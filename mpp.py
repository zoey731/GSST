# -*- coding: utf-8 -*-
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

os.environ['OPENBLAS_NUM_THREADS'] = '1'
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A5000 gpus
torch.set_float32_matmul_precision('high') # can be medium (bfloat16), high (tensorfloat32), highest (float32)

class MyDDPStrategy(strategies.DDPStrategy):
    def load_model_state_dict(self, checkpoint):
        assert self.lightning_module is not None
        self.lightning_module.load_state_dict(checkpoint["state_dict"], strict=False)

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
            mode='min' if (args.problem_type == "regression" and args.metric in['rmse','mae'] ) else 'max',
            save_on_train_epoch_end=False
        )
    )
    if args.early_stop:
        early_stop_callback = EarlyStopping( ##添加早停
        monitor=f"validation {args.metric}",  
        min_delta=0.00,                      
        patience=15,                          
        verbose=True,                        
        mode='min' if (args.problem_type == "regression" and args.metric in['rmse','mae'] ) else 'max' 
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
    elif args.mode == 'get_rep':
        trainer.test(model, ckpt_path=os.path.join(args.output_dir, args.filename, "best_model.ckpt"), datamodule=dm)
        repr = torch.cat(model.representations, dim=0)

        # 保存表征到文件
        torch.save(repr, args.root + "/representations.pt")
        print("Representations saved to representations.pt")

    else:
        raise NotImplementedError()


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
    parser.add_argument('--metric', type=str, choices=["auc", "rmse", "mae","r2"])
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

if __name__ == '__main__':
    main(get_args())