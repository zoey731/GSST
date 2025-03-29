# GSST
# Model Architecture

# Dataset
## Molecule Property Predict
We use the dataset MoleculeNet from 'MoleculeNet: A Benchmark for Molecular Machine Learning'
## Activity Cliff
We use the dataset MoleculeACE from 'Exposing the Limitations of Molecular Machine Learning with Activity Cliffs'
## Data Processing
```
python data_provider/loader.py --root_dir "your dataset root dir"  --split_mode random_scaffold  --seed 1 --task dataset_name
```
# Pretrain

# Finetune
## Train
```
python mmp.py --root 'your dataset root dir' --devices "[0]" --filename 'output dir name'  --cls_model 'cls model dir' --bert_name 'bert model dir'  --stage1_path 'pretrain weight' --max_epochs 100 --tune_gnn --precision 32 --batch_size 128 --inference_batch_size 8 --metric rmse --seed 42  --mode ft
```
## Evaluation
```
python mmp.py --root 'your dataset root dir' --devices "[0]" --filename 'output dir name'  --cls_model 'cls model dir' --bert_name 'bert model dir'  --stage1_path 'pretrain weight' --max_epochs 100 --tune_gnn --precision 32 --batch_size 128 --inference_batch_size 8 --metric rmse --seed 42  --mode test
```
