# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch_geometric.loader.dataloader import Collater
from data_provider.molecule_classify_dataset import MoleculeClassification


class TrainCollater:
    def __init__(self, tokenizer, max_length, graph_only=False):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.graph_only = graph_only
        self.collater = Collater([], [])

    def __call__(self, batch):
        graphs, smiles = zip(*batch)
        graphs = self.collater(graphs)

        smiles_tokens = self.tokenizer(
            text=smiles,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_tensors='pt',
            return_attention_mask=True
        )

        return graphs, smiles_tokens


class InferenceCollater:
    def __init__(self, tokenizer, max_length, graph_only=False):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.graph_only = graph_only
        self.collater = Collater([], [])
        
    def __call__(self, batch):
        graphs, smiles = zip(*batch)
        graphs = self.collater(graphs)

        smiles_tokens = self.tokenizer(
            text=smiles,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_tensors='pt',
            return_attention_mask=True
        )

        return graphs, smiles_tokens


class MolecularPropertyPredictionDM(LightningDataModule):
    def __init__(
        self,
        mode: str = 'ft',
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        max_length: int = 480,
        tokenizer=None,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.prompt = args.prompt
        self.graph_only = args.graph_only

        self.pretrain_dataset = MoleculeClassification(os.path.join(os.path.dirname(root), 'geometric_data_processed.pt'), self.max_length)
        self.train_dataset = MoleculeClassification(os.path.join(root, 'train.pt'), self.max_length)
        self.val_dataset = MoleculeClassification(os.path.join(root, 'dev.pt'), self.max_length)
        self.test_dataset = MoleculeClassification(os.path.join(root, 'test.pt'), self.max_length)

        self.num_labels = args.num_labels = self.pretrain_dataset.num_labels
        self.problem_type = args.problem_type = self.pretrain_dataset.problem_type
        # self.init_tokenizer(tokenizer)
        # self.mol_ph_token = '<mol>' * self.args.num_query_token

    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.pretrain_dataset.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        # self.mol_token_id = self.tokenizer.mol_token_id
        # self.tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]

    def train_dataloader(self):
        if self.mode == 'pretrain':
            loader = DataLoader(
                self.pretrain_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater(self.tokenizer, self.max_length, self.graph_only),
            )
        elif self.mode == 'ft':
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater(self.tokenizer, self.max_length, self.graph_only),
            )
        else:
            raise NotImplementedError
        return loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=TrainCollater(self.tokenizer, self.max_length, self.graph_only),
        )
        return val_loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(self.tokenizer, self.max_length, self.graph_only),
        )
        return loader
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='data/downstream_tasks')
        parser.add_argument('--max_length', type=int, default=480)
        parser.add_argument('--prompt', type=str, default=None)
        parser.add_argument('--filtered_cid_path', type=str, default=None)
        parser.add_argument('--graph_only', action='store_true', default=False)
        return parent_parser


class MolecularPropertyPredictionDM_ACE(LightningDataModule):
    def __init__(
        self,
        mode: str = 'ft',
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        max_length: int = 480,
        tokenizer=None,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.prompt = args.prompt
        self.graph_only = args.graph_only

        self.pretrain_dataset = MoleculeClassification(os.path.join(os.path.dirname(root), 'geometric_data_processed.pt'), self.max_length)
        self.train_dataset = MoleculeClassification(os.path.join(root, 'train.pt'), self.max_length)
        self.test_dataset = MoleculeClassification(os.path.join(root, 'test.pt'), self.max_length)

        self.num_labels = args.num_labels = self.pretrain_dataset.num_labels
        self.problem_type = args.problem_type = self.pretrain_dataset.problem_type
        # self.init_tokenizer(tokenizer)
        # self.mol_ph_token = '<mol>' * self.args.num_query_token

    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.pretrain_dataset.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        # self.mol_token_id = self.tokenizer.mol_token_id
        # self.tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]

    def train_dataloader(self):
        if self.mode == 'pretrain':
            loader = DataLoader(
                self.pretrain_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater(self.tokenizer, self.max_length, self.graph_only),
            )
        elif self.mode == 'ft':
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater(self.tokenizer, self.max_length, self.graph_only),
            )
        else:
            raise NotImplementedError
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(self.tokenizer, self.max_length, self.graph_only),
        )
        return loader
    

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='data/downstream_tasks')
        parser.add_argument('--max_length', type=int, default=480)
        parser.add_argument('--prompt', type=str, default=None)
        parser.add_argument('--filtered_cid_path', type=str, default=None)
        parser.add_argument('--graph_only', action='store_true', default=False)
        return parent_parser
