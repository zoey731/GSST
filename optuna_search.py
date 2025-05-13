import os
import torch
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger
from data_provider.mmp_dm import MolecularPropertyPredictionDM
from model.blip2_mmp import Blip2MMP
from model.help_funcs import rename_parameters
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import copy
import optuna
import logging

os.environ['OPENBLAS_NUM_THREADS'] = '1'
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A5000 gpus
torch.set_float32_matmul_precision('high') # can be medium (bfloat16), high (tensorfloat32), highest (float32)

class MyDDPStrategy(strategies.DDPStrategy):
    def load_model_state_dict(self, checkpoint):
        assert self.lightning_module is not None
        self.lightning_module.load_state_dict(checkpoint["state_dict"], strict=False)

def objective(trial, args):
    # 创建一份原始参数的副本，以避免修改全局参数
    args_copy = copy.deepcopy(args)
    print(args_copy)

    # 定义要优化的超参数
    init_lr = trial.suggest_loguniform('init_lr', 1e-6, 1e-4)
    batch_size = trial.suggest_categorical('batch_size', [64,128])
    max_epochs = trial.suggest_categorical('max_epochs', [100,200])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-7, 1e-3)

    # 可以继续添加更多的超参数...

    # 更新参数副本中的超参数值
    args_copy.init_lr = init_lr
    args_copy.batch_size = batch_size
    args_copy.max_epochs = max_epochs
    args_copy.weight_decay = weight_decay

    # 调用 main 函数并传递更新后的参数副本
    metric_value  = main(args_copy)
    print('-----------------------------')
    print(metric_value)

    # 返回验证集上的损失值或任何你想最小化/最大化的指标
    return metric_value 



def main(args):
    pl.seed_everything(args.seed)

    # data
    dm = MolecularPropertyPredictionDM(args.mode, args.num_workers, args.batch_size, args.root, args.max_length, None, args)

    # model
    if args.init_checkpoint:
        model = Blip2MMP.load_from_checkpoint(args.init_checkpoint, strict=False, args=args)
        print(f"loaded init checkpoint from {args.init_checkpoint}")
    elif args.stage2_path:
        model = Blip2MMP(args)
        ckpt = torch.load(args.stage2_path, weights_only=False, map_location='cpu')
        model.load_state_dict(rename_parameters(ckpt['state_dict']), strict=False)
        print(f"loaded stage2 model from {args.stage2_path}")
    elif args.stage1_path:
        model = Blip2MMP(args)
        model.load_from_stage1_checkpoint(args.stage1_path)
        print(f"loaded stage1 model from {args.stage1_path}")
    else:
        model = Blip2MMP(args)

    print('total params:', sum(p.numel() for p in model.parameters()))

    tokenizer = model.blip2cls.cls_tokenizer
    dm.init_tokenizer(tokenizer)

    callbacks = []
    ## fixme save only used parameters
    callbacks.append(
        plc.ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, args.filename),
            monitor='validation {}'.format(args.metric),
            filename='best_model',
            save_top_k=1,
            verbose=True,
            mode='min' if args.problem_type == "regression" else 'max',
            save_on_train_epoch_end=False
        )
    )
    if args.early_stop:
        early_stop_callback = EarlyStopping( ##添加早停
        monitor=f"validation {args.metric}",  
        min_delta=0.00,                      
        patience=15,                          
        verbose=True,                        
        mode='min' if args.problem_type == "regression" else 'max' 
        )
        callbacks.append(early_stop_callback)

    if len(args.devices.split(',')) > 1:
        if args.strategy_name == 'fsdp':
            strategy = strategies.DDPFullyShardedNativeStrategy()
        elif args.strategy_name == 'deepspeed':
            strategy = strategies.DeepSpeedStrategy(stage=3)
        else:
            strategy = MyDDPStrategy(find_unused_parameters=False, start_method='spawn')
    else:
        strategy = 'auto'
        args.devices = eval(args.devices)

    logger = CSVLogger(
        save_dir=os.path.join(args.output_dir, args.filename),
        # version=args.mode
    )
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=callbacks,
        strategy=strategy,
        logger=logger
    )
    if args.mode in {'pretrain', 'ft'}:
        trainer.fit(model, ckpt_path=args.ckpt_path, datamodule=dm)
        
    elif args.mode == 'validation':
        # trainer.fit_loop.epoch_progress.current.completed = args.cls_eval_epoch - 1
        trainer.validate(model, ckpt_path=os.path.join(args.output_dir, args.filename, "best_model.ckpt"), datamodule=dm)
    elif args.mode == 'test':
        trainer.test(model, ckpt_path=os.path.join(args.output_dir, args.filename, "best_model.ckpt"), datamodule=dm)
    else:
        raise NotImplementedError()
    print(trainer.callback_metrics)  # 输出所有可用的 callback_metrics
    assert 'validation {}'.format(args.metric) in trainer.callback_metrics, "Validation metric not found!"
    metric_value = trainer.callback_metrics.get('validation {}'.format(args.metric), float('inf'))
    return metric_value.item() if isinstance(metric_value, torch.Tensor) else metric_value

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="mmp")
    parser.add_argument('--output_dir', type=str, default="all_checkpoints")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # MM settings
    parser.add_argument('--mode', type=str, default='ft')
    parser.add_argument('--strategy_name', type=str, default=None)
    # parser.add_argument('--iupac_prediction', action='store_true', default=False)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--metric', type=str, choices=["auc", "rmse", "mae"])
    # parser = Trainer.add_argparse_args(parser)
    parser = Blip2MMP.add_model_specific_args(parser)  # add model args
    parser = MolecularPropertyPredictionDM.add_model_specific_args(parser)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0,1')
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--early_stop', type = bool, default=True)
    args = parser.parse_args()

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    return args

def setup_logging(filename):
    # 创建日志目录 
    log_dir = "logs"
    os.makedirs(log_dir,  exist_ok=True)
    
    # 生成日志文件路径 
    log_file = os.path.join(log_dir,  f"{filename}.log")
    
    # 配置日志 
    logging.basicConfig( 
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger() 


if __name__ == '__main__':
    args = get_args()

    logger = setup_logging(args.filename) 
    # 创建 Optuna 研究并进行优化
    if args.metric == 'rmse'or'mae':
        study = optuna.create_study(direction='minimize') 
    elif args.metric == 'auc':
        study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, args), n_trials=20)  # 运行50次试验

    print("Number of finished trials: {}".format(len(study.trials)))
    # 打印最佳试验
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    logger.info("Number  of finished trials: {}".format(len(study.trials))) 
    logger.info("Best  trial:")
    trial = study.best_trial  
    logger.info(f"   Value: {trial.value}") 
    logger.info("   Params: ")
    for key, value in trial.params.items(): 
        logger.info(f"     {key}: {value}")
    