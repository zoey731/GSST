"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from ogb.utils import smiles2graph
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import Data
import numpy as np
from lavis.models.blip2_models.blip2 import (
    # Blip2Base,
    disabled_train,
)
from model.blip2 import Blip2Base
from transformers import AutoTokenizer, AutoModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

cls_model_list = [
    "DeepChem/ChemBERTa-77M-MTR",
]


def smiles2data(smiles):
    graph = smiles2graph(smiles)
    x = torch.from_numpy(graph['node_feat'])
    edge_index = torch.from_numpy(graph['edge_index'], )
    edge_attr = torch.from_numpy(graph['edge_feat'])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


class Blip2BERT(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """
    def __init__(
        self,
        bert_name,
        gin_num_layers,
        gin_hidden_dim,
        gin_drop_ratio,
        tune_gnn=False,
        num_query_token=32,
        cross_attention_freq=2,
        tune_mode='freeze',
        peft_dir='',
        cls_model="DeepChem/ChemBERTa-77M-MTR",
        prompt="",
        args=None,
    ):
        super().__init__()
        self.args = args

        self.graph_encoder, self.ln_graph = self.init_graph_encoder(gin_num_layers, gin_hidden_dim, gin_drop_ratio)
        self.tune_gnn = tune_gnn
        if not tune_gnn:
            for name, param in self.graph_encoder.named_parameters():
                param.requires_grad = False
            self.graph_encoder = self.graph_encoder.eval()
            self.graph_encoder.train = disabled_train
            logging.info("freeze graph encoder")

        self.num_query_token = num_query_token
        self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, self.graph_encoder.num_features, cross_attention_freq)
        ### remove the unused parameters
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        ## initialize cls model
        self.cls_tokenizer = AutoTokenizer.from_pretrained(cls_model, use_fast=False, padding_side='right')
        self.cls_model = AutoModel.from_pretrained(cls_model)
        self.cls_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.cls_model.config.hidden_size
        )

        # classification head
        self.problem_type = None
        self.num_labels = args.num_labels
        self.problem_type = args.problem_type
        self.cls_head = nn.Linear(self.cls_model.config.hidden_size, args.num_labels)

        # self.collater = Collater([], [])
        self.fc = nn.Linear(300, self.cls_model.config.hidden_size)

    def forward_old(self, batch): 
        graphs, smiles_tokens = batch

        # ----------------------- 获取 2d 表征 ----------------------- #
        graph_embeds, graph_masks = self.graph_encoder(graphs)
        if not self.tune_gnn:
            graph_embeds = graph_embeds.detach()
        graph_embeds = self.ln_graph(graph_embeds, graph_masks)
        device = graph_embeds.device
        # ----------------------- 获取 2d 表征 ----------------------- #

        # ----------------------- qformer 获取 2d 信息 ----------------------- #
        query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=graph_embeds,
            encoder_attention_mask=graph_masks, # fixme: check whether this mask is correct
            return_dict=True,
        )
        query_tokens_embeds = self.cls_proj(query_output.last_hidden_state)
        # ----------------------- qformer 获取 2d 信息 ----------------------- #

        # ----------------------- 拼接 2d soft tokens 和 raw smiles tokens (embedding 层面拼接)----------------------- #
        inputs_embeds = self.cls_model.get_input_embeddings()(smiles_tokens.input_ids)
        inputs_embeds = torch.cat((query_tokens_embeds, inputs_embeds), dim=1)
        query_tokens_mask = torch.ones(query_tokens_embeds.size()[:-1], dtype=smiles_tokens.attention_mask.dtype).to(self.cls_model.device)
        attention_mask = torch.cat([query_tokens_mask, smiles_tokens.attention_mask], dim=1)
        outputs = self.cls_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        sequence_output = outputs.pooler_output
        logits = self.cls_head(sequence_output)
        loss = self.calculate_loss(logits, graphs.y)
        # ----------------------- 拼接 2d soft tokens 和 raw smiles tokens (embedding 层面拼接)----------------------- #

        return {
            "loss": loss,
            "logits": logits
        }
    
    def forward_chemberta(self, batch):## 用于验证chemberta 
        graphs, smiles_tokens = batch
        # ----------------------- 拼接 2d soft tokens 和 raw smiles tokens (embedding 层面拼接)----------------------- #
        inputs_embeds = self.cls_model.get_input_embeddings()(smiles_tokens.input_ids)

        attention_mask = torch.cat([smiles_tokens.attention_mask], dim=1)

        outputs = self.cls_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        sequence_output = outputs.pooler_output
        logits = self.cls_head(sequence_output)
        loss = self.calculate_loss(logits, graphs.y)
        # ----------------------- 拼接 2d soft tokens 和 raw smiles tokens (embedding 层面拼接)----------------------- #

        return {
            "loss": loss,
            "logits": logits
        }

    def get_repr(self, batch):
        graphs, smiles_tokens = batch

        # ----------------------- 获取 2d 表征 ----------------------- #
        graph_embeds, graph_masks = self.graph_encoder(graphs)
        if not self.tune_gnn:
            graph_embeds = graph_embeds.detach()
        graph_embeds = self.ln_graph(graph_embeds, graph_masks)
        device = graph_embeds.device
        # ----------------------- 获取 2d 表征 ----------------------- #

        # ----------------------- qformer 获取 2d 信息 ----------------------- #
        query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=graph_embeds,
            encoder_attention_mask=graph_masks, # fixme: check whether this mask is correct
            return_dict=True,
        )
        query_tokens_embeds = self.cls_proj(query_output.last_hidden_state)
        # ----------------------- qformer 获取 2d 信息 ----------------------- #

        # ----------------------- 拼接 2d soft tokens 和 raw smiles tokens (embedding 层面拼接)----------------------- #
        inputs_embeds = self.cls_model.get_input_embeddings()(smiles_tokens.input_ids)
        inputs_embeds = torch.cat((query_tokens_embeds, inputs_embeds), dim=1)
        query_tokens_mask = torch.ones(query_tokens_embeds.size()[:-1], dtype=smiles_tokens.attention_mask.dtype).to(self.cls_model.device)
        attention_mask = torch.cat([query_tokens_mask, smiles_tokens.attention_mask], dim=1)
        outputs = self.cls_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )

        mask_expanded = attention_mask.unsqueeze(-1).expand_as(inputs_embeds).float()
        # 将 mask 应用到 inputs_embeds
        masked_inputs = inputs_embeds * mask_expanded
        valid_token_counts = attention_mask.sum(dim=1, keepdim=True) 
        valid_token_counts = valid_token_counts.clamp(min=1e-9)  
        pooled_output = masked_inputs.sum(dim=1) / valid_token_counts #mean pool

        # 打印结果
        print("Pooled Output Shape:", pooled_output.shape)
        # print("Pooled Output:", pooled_output)
        return pooled_output

    
    def forward(self,batch):##用于消融实验
        graphs, smiles_tokens = batch

        # ----------------------- 获取 2d 表征 ----------------------- #
        graph_embeds, graph_masks = self.graph_encoder(graphs)
        if not self.tune_gnn:
            graph_embeds = graph_embeds.detach()
        graph_embeds = self.ln_graph(graph_embeds, graph_masks)
        device = graph_embeds.device
        # ----------------------- 获取 2d 表征 ----------------------- #
        #接一个全连接层让graph维度适应smiles
        graph_embeds = self.fc(graph_embeds)

        # ----------------------- 拼接 2d soft tokens 和 raw smiles tokens (embedding 层面拼接)----------------------- #
        inputs_embeds = self.cls_model.get_input_embeddings()(smiles_tokens.input_ids)
        inputs_embeds = torch.cat((graph_embeds, inputs_embeds), dim=1)
        graph_mask = torch.ones(graph_embeds.size()[:-1], dtype=smiles_tokens.attention_mask.dtype).to(self.cls_model.device)
        attention_mask = torch.cat([graph_mask, smiles_tokens.attention_mask], dim=1)
        if inputs_embeds.size(1) > 512:
            # 计算需要保留的smiles长度
            graph_len = graph_embeds.size(1)
            remaining_len = max(0, 512 - graph_len)  # 确保不为负
            
            # 截断inputs_embeds和attention_mask
            inputs_embeds = inputs_embeds[:, :512, :]
            attention_mask = attention_mask[:, :512]

        outputs = self.cls_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        sequence_output = outputs.pooler_output
        logits = self.cls_head(sequence_output)
        loss = self.calculate_loss(logits, graphs.y)
        # ----------------------- 拼接 2d soft tokens 和 raw smiles tokens (embedding 层面拼接)----------------------- #

        return {
            "loss": loss,
            "logits": logits
        }       

    def calculate_loss(self, logits, labels):
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            labels = labels.masked_fill(labels==-1, 0)
            if self.problem_type is None:
                if self.num_labels == 1:
                    self.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return loss