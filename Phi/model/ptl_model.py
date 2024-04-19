import lightning.pytorch as pl
from peft import get_peft_model
from transformers import AutoModelForCausalLM
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class Phi2Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.lr= config.lr
        self.eps= config.eps
        model= AutoModelForCausalLM.from_pretrained(
            config.model,
            cache_dir= config.cache_dir,
        )

        self.model = get_peft_model(model, config.peft_config)
        self.model.print_trainable_parameters()
        self.save_hyperparameters("pretrained_model")
    
    def forward(self, batch):
        outputs= self.model(
            batch['input_ids'],
            attention_mask= batch['attention_mask'], 
            labels= batch['labels']
        )
        return outputs.loss

    def training_step(self, batch):
        loss= self.forward(batch)
        self.log('train_loss', loss, logger= True, prog_bar= True, on_step=True)
        return loss
    
    def configure_optimizers(self):
        optim  = AdamW(
            params=self.trainer.model.parameters(),
            eps= self.eps,
            lr=self.lr,
            weight_decay=0.0,
        )
        
        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer=optim,
            T_0 = 15,
            T_mult = 1,
            eta_min = 8e-6,
            verbose = True
        )
        
        return {'optimizer': optim, 'lr_scheduler': lr_scheduler} 