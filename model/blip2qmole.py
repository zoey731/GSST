"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license smiles, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import os
import torch
import torch.distributed as dist
from torch import Tensor
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from typing import Any, Iterable, Iterator, List, Optional, Sized, Tuple, Union, Dict
from model.help_funcs import pad_and_concat

# from lavis.common.registry import registry
# from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput
from lavis.common.dist_utils import is_dist_avail_and_initialized
from model.blip2 import Blip2Base
# from pytorch_lightning.utilities import distributed

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    print('running here')
    return output

# @torch.no_grad()
# def pl_concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     # if use distributed training
#     if not is_dist_avail_and_initialized():
#         return tensor

#     tensors_gather = distributed.gather_all_tensors(tensor)
#     output = torch.cat(tensors_gather, dim=0)
#     return output

@torch.no_grad()
def pl_concat_all_gather(tensor, padding=False, fill_value=0):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = gather_all_tensors(tensor)
    if padding:
        output = pad_and_concat(tensors_gather, fill_value=fill_value).detach()
    else:
        output = torch.cat(tensors_gather, dim=0)
    return output


def gather_all_tensors(*args: Any, **kwargs: Any) -> Any:
    return _gather_all_tensors(*args, **kwargs)

def _gather_all_tensors(result: Tensor, group: Optional[Any] = None) -> List[Tensor]:
    """Function to gather all tensors from several DDP processes onto a list that is broadcasted to all processes.

    Works on tensors that have the same number of dimensions, but where each dimension may differ. In this case
    tensors are padded, gathered and then trimmed to secure equal workload for all processes.

    Args:
        result: The value to sync
        group: The process group to gather results from. Defaults to all processes (world)

    Return:
        gathered_result: List with size equal to the process group where
            gathered_result[i] corresponds to result tensor from process i
    """
    if group is None:
        group = torch.distributed.group.WORLD

    # Convert tensors to contiguous format
    result = result.contiguous()

    world_size = torch.distributed.get_world_size(group)
    torch.distributed.barrier(group=group)

    # If the tensor is scalar, things are easy
    if result.ndim == 0:
        return _simple_gather_all_tensors(result, group, world_size)

    # 1. Gather sizes of all tensors
    local_size = torch.tensor(result.shape, device=result.device)
    local_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(local_sizes, local_size, group=group)
    max_size = torch.stack(local_sizes).max(dim=0).values
    all_sizes_equal = all(all(ls == max_size) for ls in local_sizes)

    # 2. If shapes are all the same, then do a simple gather:
    if all_sizes_equal:
        return _simple_gather_all_tensors(result, group, world_size)

    # 3. If not, we need to pad each local tensor to maximum size, gather and then truncate
    pad_dims = []
    pad_by = (max_size - local_size).detach().cpu()
    for val in reversed(pad_by):
        pad_dims.append(0)
        pad_dims.append(val.item())
    result_padded = F.pad(result, pad_dims)
    gathered_result = [torch.zeros_like(result_padded) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_result, result_padded, group)
    for idx, item_size in enumerate(local_sizes):
        slice_param = [slice(dim_size) for dim_size in item_size]
        gathered_result[idx] = gathered_result[idx][slice_param]
    return gathered_result

def _simple_gather_all_tensors(result: Tensor, group: Any, world_size: int) -> List[Tensor]:
    gathered_result = [torch.zeros_like(result) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_result, result, group)
    return gathered_result

# @registry.register_model("blip2")
# @registry.register_model("blip2_feature_extractor")
class Blip2QMole(Blip2Base):
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
        gtm,
        lm,
        bert_name,
        temperature,
        gin_num_layers,
        gin_hidden_dim,
        gin_drop_ratio,
        tune_gnn=False,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
    ):
        super().__init__()
        self.gtm = gtm
        self.lm = lm
        
        self.tokenizer = self.init_tokenizer(bert_name)

        self.graph_encoder, self.ln_graph = self.init_graph_encoder(gin_num_layers, gin_hidden_dim, gin_drop_ratio)
        self.tune_gnn = tune_gnn
        if not tune_gnn:
            for name, param in self.graph_encoder.named_parameters():
                param.requires_grad = False
            self.graph_encoder = self.graph_encoder.eval()
            self.graph_encoder.train = disabled_train
            logging.info("freeze graph encoder")
        
        self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, self.graph_encoder.num_features, cross_attention_freq)
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.graph_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.smiles_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.gtm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temperature = temperature

    def contrast_global(self, features_graph, features_smiles, features_graph_all, features_smiles_all, return_sim=False):
        '''
        features_graph: shape = [B, num_qs, D]
        features_smiles: shape = [B, D]
        features_smiles_all: shape = [B * num_gpus, D]
        features_graph_all: shape = [B * num_gpus, num_qs, D]
        '''
        bs = features_graph.size(0)

        # cosine similarity as logits
        sim_q2t = (features_graph.unsqueeze(1) @ features_smiles_all.unsqueeze(-1)).squeeze() # shape = [B, 1, num_qs, D]; shape = [B * num_gpus, D, 1]; output shape = [B, B * num_gpus, num_qs]
        sim_g2t, _ = sim_q2t.max(-1) # shape = [B, B * num_gpus]

        logits_per_graph = sim_g2t / self.temperature
    

        sim_t2q = (features_smiles.unsqueeze(1).unsqueeze(1) @ features_graph_all.permute(0, 2, 1)).squeeze() # shape = [B, 1, 1, D]; [B*num_gpus, D, num_qs]; output shape = [B, B*num_gpus, 1, num_qs]
        sim_t2g, _ = sim_t2q.max(-1)
        logits_per_smiles = sim_t2g / self.temperature

        # labels = torch.arange(bs, dtype=torch.long, device=self.device)
        try:
            rank = dist.get_rank()
        except Exception:
            rank = 0
        labels = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(self.device)

        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_smiles = F.cross_entropy(logits_per_smiles, labels)
        loss = (loss_graph + loss_smiles) / 2

        if return_sim:
            return logits_per_graph[:, rank*bs:rank*bs+bs], logits_per_smiles[:, rank*bs:rank*bs+bs], loss
        else:
            return loss

    def forward(self, batch):
        ## v2: gather results from all gpus
        ###============== Image-smiles Contrastive ===================###
        graph, smiles, mask = batch
        batch_node, batch_mask = self.graph_encoder(graph)
        if not self.tune_gnn:
            batch_node = batch_node.detach()
        batch_size = batch_node.shape[0]

        batch_node = self.ln_graph(batch_node, batch_mask)
        query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=batch_node,
            encoder_attention_mask=batch_mask, # fixme: check whether this mask is correct
            use_cache=True,
            return_dict=True,
        )
        graph_feats = self.graph_proj(query_output.last_hidden_state) # shape = [B, num_q, D]
        smiles_output = self.Qformer.bert(smiles, attention_mask=mask, return_dict=True) # shape = [B, n_max, D]
        smiles_feats = self.smiles_proj(smiles_output.last_hidden_state[:, 0, :])
        
        smiles_feats, graph_feats = F.normalize(smiles_feats, p=2, dim=-1), F.normalize(graph_feats, p=2, dim=-1)
        smiles_feats_all, graph_feats_all = pl_concat_all_gather(smiles_feats), pl_concat_all_gather(graph_feats) # shape = [B * num_gpus, D]
        sim_g2t, sim_t2g, loss_gtc = self.contrast_global(graph_feats, smiles_feats, graph_feats_all, smiles_feats_all, return_sim=True)


        ###============== Image-smiles Matching ===================###
        loss_gtm = 0
        if self.gtm:
            ## not aggregate global tensor because of their different shapes
            g_emb_world = batch_node
            g_mask_world = batch_mask
            smiles_ids_world = smiles
            smiles_mask_world = mask
            with torch.no_grad():
                weights_t2g = F.softmax(sim_t2g, dim=1) + 1e-4
                weights_t2g.fill_diagonal_(0)
                weights_g2t = F.softmax(sim_g2t, dim=1) + 1e-4
                weights_g2t.fill_diagonal_(0)

            # select a negative graph for each smiles
            graph_embeds_neg = []
            graph_mask_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_t2g[b], 1).item()
                graph_embeds_neg.append(g_emb_world[neg_idx])
                graph_mask_neg.append(g_mask_world[neg_idx])
            
            graph_embeds_neg = torch.stack(graph_embeds_neg, dim=0)
            graph_mask_neg = torch.stack(graph_mask_neg, dim=0)

            # select a negative smiles for each image
            smiles_ids_neg = []
            smiles_atts_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_g2t[b], 1).item()
                smiles_ids_neg.append(smiles_ids_world[neg_idx])
                smiles_atts_neg.append(smiles_mask_world[neg_idx])

            smiles_ids_neg = torch.stack(smiles_ids_neg, dim=0)
            smiles_atts_neg = torch.stack(smiles_atts_neg, dim=0)

            smiles_ids_all = torch.cat(
                [smiles, smiles, smiles_ids_neg], dim=0
            )  # pos, pos, neg
            smiles_atts_all = torch.cat(
                [mask, mask, smiles_atts_neg],
                dim=0,
            )

            query_tokens_itm = self.query_tokens.expand(smiles_ids_all.shape[0], -1, -1)
            query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long, device=smiles.device)
            attention_mask_all = torch.cat([query_atts_itm, smiles_atts_all], dim=1)

            graph_embeds_all = torch.cat([batch_node, graph_embeds_neg, batch_node], dim=0)  # pos, neg, pos
            graph_atts_all = torch.cat([batch_mask, graph_mask_neg, batch_mask], dim=0)

            output_itm = self.Qformer.bert(
                smiles_ids_all,
                query_embeds=query_tokens_itm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=graph_embeds_all,
                encoder_attention_mask=graph_atts_all,
                return_dict=True,
            )

            vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :] # keep query tokens only
            vl_output = self.gtm_head(vl_embeddings)
            logits = vl_output.mean(dim=1)

            itm_labels = torch.cat(
                [torch.ones(batch_size, dtype=torch.long), torch.zeros(2 * batch_size, dtype=torch.long)],
                dim=0,
            ).to(smiles.device)
            loss_gtm = F.cross_entropy(logits, itm_labels)

        return BlipOutput(
            loss=loss_gtc + loss_gtm,
            loss_itc=loss_gtc,
            loss_itm=loss_gtm,
        )
    
    def graph_forward(self, graph):
        batch_node, batch_mask = self.graph_encoder(graph)
        batch_node = self.ln_graph(batch_node, batch_mask)
        query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=batch_node,
            encoder_attention_mask=batch_mask, # fixme: check whether this mask is correct
            use_cache=False,
            return_dict=True,
        )
        graph_feats = self.graph_proj(query_output.last_hidden_state) # shape = [B, num_q, D]
        graph_feats = F.normalize(graph_feats, p=2, dim=-1)
        return graph_feats, batch_node, batch_mask

    def smiles_forward(self, smiles, mask):
        smiles_output = self.Qformer.bert(smiles, attention_mask=mask, return_dict=True) # shape = [B, n_max, D]
        smiles_feats = self.smiles_proj(smiles_output.last_hidden_state[:, 0, :] )
        smiles_feats = F.normalize(smiles_feats, dim=-1, p=2)
        return smiles_feats
    
    def compute_gtm(self, batch_node, batch_mask, smiles_ids, smiles_atts):
        '''
        batch_node shape = [B, N, D]
        batch_mask shape = [B, N]
        smiles_ids shape = [B, N]
        smiles_atts shape = [B, N]
        '''
        query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1) # shape = [B, Nq, D]
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            batch_node.device
        ) # shape = [B, Nq]
        attention_mask = torch.cat([query_atts, smiles_atts], dim=1) # shape = [B, Nq + N]
        output_gtm = self.Qformer.bert(
            smiles_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=batch_node,
            encoder_attention_mask=batch_mask,
            return_dict=True,
        )
        gl_embeddings = output_gtm.last_hidden_state[:, : query_tokens.size(1), :] # shape = [B, Nq, D]
        gtm_logit = self.gtm_head(gl_embeddings).mean(dim=1) # shape = [B, Nq, 2]
        # gtm_logit = F.softmax(gtm_logit, dim=-1)[:, 1] # select the axis of the positive class
        gtm_logit = gtm_logit[:, 1] # select the axis of the positive class
        return gtm_logit
