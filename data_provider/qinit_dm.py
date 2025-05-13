# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import torch_geometric
from pytorch_lightning import LightningDataModule
from torch_geometric.data import Batch
from data_provider.pretrain_dataset import GINPretrainDataset
from data_provider.retrieval_dataset import RetrievalDataset
from data_provider.molecule_classify_dataset import MoleculeClassification
from torch.utils.data import DataLoader


class TrainCollater(object):
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch):
        data_list, smiles = zip(*batch)
        graph_batch = Batch.from_data_list(data_list)        
        smiles_batch = self.tokenizer(smiles, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        return graph_batch, smiles_batch.input_ids, smiles_batch.attention_mask

# class TrainCollater(object):
#     def __init__(self, tokenizer, max_length):
#         self.tokenizer = tokenizer
#         self.max_length = max_length
    
#     def __call__(self, batch):
#         data_list, smiles_list, smiles_prompt_list = zip(*batch)
#         graph_batch = Batch.from_data_list(data_list[0], data_list)        
#         smiles_batch = self.tokenizer(smiles_list, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
#         return graph_batch, smiles_batch.input_ids, smiles_batch.attention_mask


class QInitDM(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        max_length: int = 128,
        graph_aug: str = 'dnodes',
        tokenizer=None,
        args=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.match_batch_size = args.match_batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.max_length = max_length
        print('Loading PubChem324k dataset')
        self.train_dataset = MoleculeClassification(os.path.join(os.path.dirname(root), 'pretrain.pt'), self.max_length)
        self.val_dataset = MoleculeClassification(os.path.join(os.path.dirname(root), 'valid.pt'), self.max_length)
        self.val_dataset_match = MoleculeClassification(os.path.join(os.path.dirname(root), 'valid.pt'), self.max_length).shuffle()
        self.test_dataset_match = MoleculeClassification(os.path.join(os.path.dirname(root), 'test.pt'), self.max_length).shuffle()
        self.val_match_loader = DataLoader(self.val_dataset_match, batch_size=self.match_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False, drop_last=False, persistent_workers=True, collate_fn=TrainCollater(self.tokenizer, max_length))
        self.test_match_loader = DataLoader(self.test_dataset_match, batch_size=self.match_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False, drop_last=False, persistent_workers=True, collate_fn=TrainCollater(self.tokenizer, max_length))

    def train_dataloader(self):
        if False:
            loader = torch_geometric.loader.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True
            )
        else:
            loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=False, drop_last=True, persistent_workers=True, collate_fn=TrainCollater(self.tokenizer, self.max_length))
        # print('len(train_dataloader)', len(loader))
        return loader

    def val_dataloader(self):
        if False:
            loader = torch_geometric.loader.DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=False,
                persistent_workers=True
            )
        else:
            loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False, drop_last=False, persistent_workers=True, collate_fn=TrainCollater(self.tokenizer, self.max_length))
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--match_batch_size', type=int, default=64)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='data/PubChemDataset/PubChem-320k')
        parser.add_argument('--max_length', type=int, default=480)
        parser.add_argument('--graph_aug', type=str, default='dnodes')
        parser.add_argument('--smiles_aug', action='store_true', default=False)
        parser.add_argument('--use_phy_eval', action='store_true', default=False)
        parser.add_argument('--filtered_cid_path', type=str, default=None)
        return parent_parser