import os
from functools import partial

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
from transformers.models.phi.modeling_phi import PhiDecoderLayer

from ray.train import Checkpoint
from ray.train.lightning import RayFSDPStrategy

from dataclasses import dataclass
from peft import LoraConfig, TaskType

#phi2
MODEL_NAME= 'microsoft/phi-2'
DS_PATH = Checkpoint(os.path.join(os.path.dirname(__file__), '../synthetic_ds.pq'))
CHECKPOINT= Checkpoint(os.path.join(os.path.dirname(__file__), '../checkpoints'))

PEFT_CONFIG= LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules= ['q_proj', 'k_proj', 'v_proj'],
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias = 'none',
)

AUTO_WRAP_POLICY= partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls = {PhiDecoderLayer}
)

FSDP= RayFSDPStrategy(
    sharding_strategy= ShardingStrategy.FULL_SHARD,
    backward_prefetch= BackwardPrefetch.BACKWARD_POST,
    forward_prefetch= True,
    auto_wrap_policy= AUTO_WRAP_POLICY,
    limit_all_gathers=True,
    activation_checkpointing= PhiDecoderLayer
)

@dataclass
class TrainConfig:
    model: str= MODEL_NAME
    peft_config: LoraConfig= PEFT_CONFIG
    ds_path: str= DS_PATH
    ckpt: Checkpoint= CHECKPOINT
    strategy: RayFSDPStrategy= FSDP
    num_worker: int= 1
    batch_size_per_worker: int= 2
    resources: dict= {'GPU': 1}
    log_every_n_steps: int= 1
    devices: str= 'auto'
    acclerator: str= 'auto'
    percision: str= '16-mixed'
    max_epochs: int= 10
    eps: float= 1e-8
    lr: float= 5e-3
