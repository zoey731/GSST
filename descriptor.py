# -*- coding: utf-8 -*-
import torch
from torch_geometric.data import InMemoryDataset

import sys
sys.path.append("..")

import numpy as np
from multiprocessing import Pool
from rdkit import Chem
from scipy import sparse as sp
import argparse 
import sys
sys.path.append("..")
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
from rdkit import Chem
from scipy import sparse as sp
import argparse 

import numpy as np  
from scipy.sparse  import csc_matrix  
import torch  
from torch_geometric.data  import InMemoryDataset, Data  

from kpgt.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized

    
class PubChemDataset(InMemoryDataset):  
    def __init__(self, path):  
        super(PubChemDataset, self).__init__()  
        self.data,  self.slices  = torch.load(path)   
 
    def __getitem__(self, idx):  
        return self.get(idx)   
 
    # 新增特征融合方法  
    def fuse_features(self, fp_matrix, md_matrix):  
        """  
        fp_matrix: numpy数组形式的分子指纹矩阵  
        md_matrix: numpy数组形式的分子描述符矩阵  
        """  
        # 步骤1：深拷贝Data对象列表  
        new_data_list = [data.clone() for data in self]  
 
        # 步骤2：添加新特征  
        for i, data in enumerate(new_data_list):  
            data.fp  = torch.from_numpy(fp_matrix[i]).float()   
            data.md  = torch.from_numpy(md_matrix[i]).float()   
 
        # 步骤3：重建存储结构  
        self.data,  self.slices  = self.collate(new_data_list)   
 
        # 步骤4：持久化到磁盘
        torch.save((self.data,  self.slices),  args.output_pth_path)   
    
def preprocess_dataset(args):
    # 加载预训练数据集  
    dataset = PubChemDataset(args.root)   
    smiles_data = []
    # 提取SMILES数据（优化为列表推导式）
    for i in range(len(dataset)):
        print(dataset[i].smiles)
        smiles_data.append(dataset[i].smiles)
    ##已有smiles
    print('extracting fingerprints')
    FP_list = []
    for smiles in smiles_data:
        mol = Chem.MolFromSmiles(smiles)
        FP_list.append(list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)))
        a = list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512))
        assert not np.isnan(a).any(),"fp有空值!!!!!"
    FP_arr = np.array(FP_list)
    # print('-------------')
    # print(FP_arr.shape)
    # FP_sp_mat = sp.csc_matrix(FP_arr)##存储为稀疏矩阵
    # print('saving fingerprints')
    # sp.save_npz(f"{OUTPUT}/fp.npz", FP_sp_mat)
    print('fp已经处理完毕')


    print('extracting molecular descriptors')
    generator = RDKit2DNormalized()
    features_map = Pool(args.n_jobs).imap(generator.process, smiles_data)
    arr = np.array(list(features_map))

    arr = np.nan_to_num(np.array(list(arr)),  nan=0.0)

    assert not np.isnan(arr[:,1:]).any(),"md有空值!!!!"
    md=arr[:,1:]
    # np.savez_compressed(f"{OUTPUT}/md.npz",md=arr[:,1:])
    print('md已经处理完毕')
    # print(md.shape)



    dataset = PubChemDataset(args.root)
    
    # 加载分子指纹和描述符 
    # with np.load(NPZ_MD_PATH)  as npz1, np.load(NPZ_FP_PATH)  as npz2: 
    # fp = csc_matrix(  
    #     (FP_sp_mat['data'], FP_sp_mat['indices'], FP_sp_mat['indptr']),  
    #     shape=FP_sp_mat['shape']).toarray() 

    #     md = npz1['md']  
    fp =FP_arr
    dataset.fuse_features(fp,  md)

    print("done")


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--output_pth_path", type=str, required=True)
    parser.add_argument("--path_length", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=32)
    args = parser.parse_args()
    return args


# ROOT = '/home/zhoujie/MolCA/data/descriptors/bace/processed/random_scaffold_42/test.pt'  
# OUTPUT_PT_PATH = "/home/zhoujie/MolCA/data/descriptors/bace/processed/test.pt"   
# OUTPUT = "/home/zhoujie/MolCA//data/descriptors/bace/" 


if __name__ == '__main__':
    args = parse_args()
    preprocess_dataset(args)