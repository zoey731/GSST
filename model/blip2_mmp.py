import os
import torch
import json
import numpy as np
import torch.distributed as dist
import pytorch_lightning as pl
from torch import optim
from typing import Any, Dict
from transformers import Adafactor
from model.blip2_bert import Blip2BERT
from model.help_funcs import caption_evaluate, AttrDict, eval_rocauc, eval_rmse, eval_mae,eval_r2
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler


def load_ignore_unexpected(model, state_dict):
    keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}
    
    ## try to print keys that are not included
    model.load_state_dict(state_dict, strict=True)


def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1:]
            if key == '':
                return value
            module_state_dict[key] = value
    return module_state_dict


# peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
class Blip2MMP(pl.LightningModule):
    # def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    #     # checkpoint.pop('optimizer_states')
    #     to_be_removed = []
    #     for key, value in checkpoint['state_dict'].items():
    #         try:
    #             if not self.get_parameter(key).requires_grad:
    #                 to_be_removed.append(key)
    #         except AttributeError:
    #             to_be_removed.append(key)
    #     for key in to_be_removed:
    #         checkpoint['state_dict'].pop(key)

    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        # self.cls_eval_epoch = args.cls_eval_epoch
        # self.reaction_weight = args.reaction_weight
        self.tune_mode = args.tune_mode
        if args.cls_model.find('ChemBERTa') >= 0:
            self.blip2cls = Blip2BERT(args.bert_name, args.gin_num_layers, args.gin_hidden_dim, args.drop_ratio, args.tune_gnn, args.num_query_token, args.cross_attention_freq, args.tune_mode, args.peft_dir, args.cls_model, args.prompt, args)
        elif args.cls_model.find('bert') >= 0: ##
            self.blip2cls = Blip2BERT(args.bert_name, args.gin_num_layers, args.gin_hidden_dim, args.drop_ratio, args.tune_gnn, args.num_query_token, args.cross_attention_freq, args.tune_mode, args.peft_dir, args.cls_model, args.prompt, args)
        else:
            raise NotImplementedError()
        self.tokenizer = self.blip2cls.init_tokenizer(args.bert_name)
        self.save_hyperparameters(args)
        ##用于存储表征
        self.representations = []

    def load_from_stage1_checkpoint(self, path):
        ckpt = torch.load(path, weights_only=False, map_location='cpu')
        state_dict = ckpt['state_dict']
        graph_encoder_dict = get_module_state_dict(state_dict, 'blip2qformer.graph_encoder')
        qformer_dict = get_module_state_dict(state_dict, 'blip2qformer.Qformer')
        ln_graph_dict = get_module_state_dict(state_dict, 'blip2qformer.ln_graph')
        qs_weight = get_module_state_dict(state_dict, 'blip2qformer.query_tokens')
        load_ignore_unexpected(self.blip2cls.Qformer, qformer_dict)
        self.blip2cls.graph_encoder.load_state_dict(graph_encoder_dict)
        self.blip2cls.ln_graph.load_state_dict(ln_graph_dict)
        self.blip2cls.query_tokens.data.copy_(qs_weight)
        return self

    def configure_optimizers(self):
        if self.args.optimizer == 'adafactor':
            print('Using adafactor optimizer')
            optimizer = Adafactor(
                self.parameters(),
                lr=1e-3,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False
            )
            self.scheduler = None
        else:
            self.trainer.fit_loop.setup_data()
            warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)
            optimizer = optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
            if self.args.scheduler == 'linear_warmup_cosine_lr':
                self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, warmup_steps, self.args.warmup_lr)
            elif self.args.scheduler == 'linear_warmup_step_lr':
                self.scheduler = LinearWarmupStepLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, self.args.lr_decay_rate, self.args.warmup_lr, warmup_steps)
            elif self.args.scheduler == 'None':
                self.scheduler = None
            else: 
                raise NotImplementedError()
        return optimizer

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        loss = self.blip2cls(batch)
        batch_size = batch[-1].input_ids.size(0)
        self.log("Training loss", float(loss['loss']), batch_size=batch_size, prog_bar=True, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, prog_bar=True, sync_dist=True)
        return loss['loss']

    # def on_train_epoch_end(self):
    #     self.trainer.test(self)

    def on_validation_epoch_start(self) -> None:
        self.list_predictions = []
        self.list_targets = []

    @torch.no_grad()
    def validation_step(self, batch):
        graphs, smiles_tokens = batch
        batch_size = smiles_tokens.input_ids.shape[0]
        outputs = self.blip2cls(batch)
        self.list_predictions.append(outputs["logits"])
        self.list_targets.append(graphs.y)
        ###============== Overall Loss ===================###
        self.log("validation loss", float(outputs['loss']), batch_size=batch_size, on_step=False, on_epoch=True, sync_dist=True)
        return outputs['loss']
    
    def on_validation_epoch_end(self) -> None:
        self.calculate_metrics(phrase="validation")

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        graphs, smiles_tokens = batch
        predictions = self.blip2cls(batch)

        representation = self.blip2cls.get_repr(batch)
        self.representations.append(representation.detach().cpu())  # 保存表征


        self.list_predictions.append(predictions["logits"])
        self.list_targets.append(graphs.y)
        return predictions["logits"], graphs.y

    def on_test_epoch_end(self):
        self.calculate_metrics(phrase="test")

    def calculate_metrics(self, phrase="validation"):
        list_predictions = self.list_predictions
        list_targets = self.list_targets
        predictions = [i for ii in list_predictions for i in ii]
        targets = [i for ii in list_targets for i in ii]

        all_predictions = [None for _ in range(self.trainer.world_size)]
        all_targets = [None for _ in range(self.trainer.world_size)]
        try:
            dist.all_gather_object(all_predictions, predictions)
            dist.all_gather_object(all_targets, targets)
        except (RuntimeError, ValueError):
            all_predictions = [predictions]
            all_targets = [targets]

        if self.global_rank == 0:
            all_predictions = torch.stack([i.detach().cpu() for ii in all_predictions for i in ii], dim=0)
            all_targets = torch.stack([i.detach().cpu() for ii in all_targets for i in ii], dim=0)
            if self.args.problem_type == "regression":
                assert self.args.metric in ["rmse", "mae","r2"], "regression only support rmse or mae or r2 metric."
                if self.args.metric == "rmse":
                    metric = eval_rmse(all_predictions.numpy(), all_targets.numpy())
                elif self.args.metric == "mae":
                    metric = eval_mae(all_predictions.numpy(), all_targets.numpy())
                else:
                    metric = eval_r2(all_predictions.numpy(),all_targets.numpy())
                self.log("{} {}".format(phrase, self.args.metric), metric, sync_dist=True)
            elif self.args.num_labels > 1 and (all_targets.dtype == torch.long or all_targets.dtype == torch.int):
                assert self.args.metric == "auc", "single label classification only support auc metric."
                auc = eval_rocauc(all_predictions[:, 1].unsqueeze(-1).numpy(), all_targets.numpy())
                self.log("{} auc".format(phrase), auc, sync_dist=True)
                all_predictions = all_predictions.argmax(-1)
            else:
                assert self.args.metric == "auc", "multi-label classification only support auc metric."
                auc = eval_rocauc(all_predictions.numpy(), all_targets.numpy())
                self.log("{} auc".format(phrase), auc, sync_dist=True)
                all_predictions = torch.sigmoid(all_predictions)
            self.save_predictions(all_predictions.numpy().squeeze(), all_targets.numpy().squeeze())

    def save_predictions(self, predictions, targets):
        assert len(predictions) == len(targets) and type(predictions) == type(targets)
        if isinstance(predictions, np.ndarray) and isinstance(targets, np.ndarray):
            predictions, targets = predictions.tolist(), targets.tolist()
        with open(os.path.join(self.logger.log_dir, 'predictions.txt'), 'w', encoding='utf8') as f:
            for p, t in zip(predictions, targets):
                line = {'prediction': p, 'target': t}
                f.write(json.dumps(line, ensure_ascii=True) + '\n')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GINSimclr")
        # train mode
        # GIN
        parser.add_argument('--gin_hidden_dim', type=int, default=300)
        parser.add_argument('--gin_num_layers', type=int, default=5)
        parser.add_argument('--drop_ratio', type=float, default=0.0)
        parser.add_argument('--tune_gnn', action='store_true', default=False)
        # Bert
        parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
        parser.add_argument('--bert_name', type=str, default='/home/zhoujie/pretrained_models/scibert_scivocab_uncased')
        parser.add_argument('--cross_attention_freq', type=int, default=2)
        parser.add_argument('--num_query_token', type=int, default=8)
        # CLS
        parser.add_argument('--cls_model', type=str, default="/home/zhoujie/pretrained_models/ChemBERTa-77M-MLM")

        parser.add_argument('--peft_dir', type=str, default='')
        parser.add_argument('--peft_config', type=str, default=None)
        parser.add_argument('--tune_mode', type=str, default=None)
        parser.add_argument('--save_every_n_epochs', type=int, default=10)
        ## quantization
        parser.add_argument('--load_in_8bit', action='store_true', default=False)

        ## lora config
        parser.add_argument('--lora_r', type=int, default=8)
        parser.add_argument('--lora_alpha', type=int, default=32)
        parser.add_argument('--lora_dropout', type=int, default=0.1)

        # optimization
        parser.add_argument('--reaction_weight', type=float, default=1.0)
        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=1e-5, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=200, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler') # or linear_warmup_step_lr
        parser.add_argument('--optimizer', type=str, default='adamw', help='type of scheduler')
        parser.add_argument('--stage1_path', type=str, default='')
        parser.add_argument('--stage2_path', type=str, default='')
        parser.add_argument('--init_checkpoint', type=str, default='')
        # parser.add_argument('--cls_eval_epoch', type=int, default=1, help='control the test phase frequency (from MolCA)')
        return parent_parser