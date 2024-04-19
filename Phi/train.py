import lightning.pytorch as pl
from datasets import load_dataset

import ray
from ray.train.torch import TorchTrainer
from ray.train import (
    ScalingConfig,
    RunConfig,
    CheckpointConfig
)
from ray.train.lightning import (
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer
)

from model import Phi2Model
from config import TrainConfig

##########################################################################

def train_func(
    config: TrainConfig
)-> None:
    
    model= Phi2Model(config)
    train_ds=  ray.train.get_dataset_shard("train")
    train_dataloader= train_ds.iter_torch_batches(
        batch_size= config.batch_size_per_worker
    )

    trainer= pl.Trainer(
        max_epochs=config.max_epochs,
        devices= config.devices,
        accelerator= config.acclerator,
        precision= config.acclerator,
        strategy= config.strategy,
        plugins= [RayLightningEnvironment()],
        callbacks= [RayTrainReportCallback()],
        enable_checkpointing= False,
    )

    trainer= prepare_trainer(trainer)
    
    trainer.fit(
        model,
        train_dataloader= train_dataloader
    )

##########################################################################

train_config= TrainConfig()
run_config= RunConfig(
    name= 'finetune_phi-v2-2.7b',
    storage_path=train_config.ckpt.path,
    storage_filesystem=train_config.ckpt.filesystem,
    checkpoint_config=CheckpointConfig(num_to_keep= 3)
)
scaling_config= ScalingConfig(
    num_workers= train_config.num_worker,
    resources_per_worker= train_config.resources,
    trainer_resources= train_config.resources,
    use_gpu= True,
)

train_ds= load_dataset(train_config.ds_path)['train']

ray_trainer = TorchTrainer(
    train_func,
    train_loop_config=train_config,
    run_config=run_config,
    scaling_config=scaling_config,
    datasets={"train": train_ds},
)
result = ray_trainer.fit()
result