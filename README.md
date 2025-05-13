# GSST: Multimodal Graph-SMILES Fusion with Soft SMILES Tokens for  Molecular Property Prediction
Code repository for paper "GSST: Multimodal Graph-SMILES Fusion with Soft SMILES Tokens for  Molecular Property Prediction"
## Model Architecture
![GSST](https://github.com/user-attachments/assets/e150234b-e15f-431e-a8b6-dad8b6e17ff6)

## Setup and dependencies
 Python 3.10
- rdkit==2022.3.3
- torch==2.2.2+cu118
## Dataset
- Molecule Property Predict: We use the dataset MoleculeNet from 'MoleculeNet: A Benchmark for Molecular Machine Learning'
- Activity Cliff: We use the dataset MoleculeACE from 'Exposing the Limitations of Molecular Machine Learning with Activity Cliffs'
### Data Processing
```
python data_provider/loader.py \
  --root_dir "your dataset root dir"  \
  --split_mode random_scaffold  \
  --seed 1 \
  --task datasetname
```
EXAMPLE:
```
data/
  - downstream_tasks/
    - bace/
      - processed/
        - {SPLIT_MODE}_{SEED}/
      - raw/
        - bace.csv
```

## Pretrain
You can pre-train your model using the following :
```
python qinit.py
  --root data/PubChem324kV2/
  --devices "[0]"
  --filename qinit_chemberta_32
  --mode train
  --rerank_cand_num 128
  --num_query_token 32
  --batch_size 64
  --accumulate_grad_batches 4
  --tune_gnn
  --gtm
  --lm
  --bert_name /pretrained_models/ChemBERTa-77M-MLM
```
Or we provide our weights in 
## Finetune
### Train
```
python mpp.py \
  --root 'your dataset root dir' \
  --devices "[0]" \
  --filename 'output dir name' \
  --cls_model 'cls model dir' \
  --bert_name 'bert model dir'  \
  --stage1_path 'pretrain weight' \
  --max_epochs 100 \
  --max_length 200 \
  --tune_gnn \
  --precision 32 \
  --batch_size 128 \
  --inference_batch_size 8 \
  --metric rmse \
  --seed 42  \
  --mode ft
```
### Evaluation
```
python mpp.py \
  --root 'your dataset root dir' \
  --devices "[0]" \
  --filename 'output dir name' \
  --cls_model /pretrained_models/ChemBERTa-77M-MLM \
  --bert_name /pretrained_models/ChemBERTa-77M-MLM  \
  --stage1_path 'pretrain weight' \
  --max_epochs 100 \
  --max_length 200 \
  --tune_gnn \
  --precision 32 \
  --batch_size 128 \
  --inference_batch_size 8 \
  --metric rmse \
  --seed 42  \
  --mode test
```
