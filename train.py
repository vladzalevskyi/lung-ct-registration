import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from dataset import LungDatasets
from pathlib import Path
import nibabel as nib
import csv
from models import VoxelMorphPP
# use the configuration for the dataloaders
def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == '__main__':
    train_loader = DataLoader(LungDatasets(['learn2reg'],
                                           Path('data'),
                                           partitions=['train'],
                                           transform=None,
                                           ),
                              batch_size=1, # keep as 1 since the code is bad
                              num_workers=8,) 
    # trainer = pl.Trainer(max_epochs=1,
    #                      log_every_n_steps=5)
    model = VoxelMorphPP.load_from_checkpoint('lightning_logs/version_1/checkpoints/epoch=0-step=16.ckpt')

    # trainer.fit(model, train_dataloaders=train_loader)
    
    # call validation
    # call after training
    # automatically auto-loads the best weights from the previous run
    # trainer.test(dataloaders=test_dataloader)

    # # or call with pretrained model
    # model = VoxelMorphPP.load_from_checkpoint(PATH)
    # trainer = Trainer()
    # trainer.test(model, dataloaders=test_dataloader)
    
    # call predict
    
    # # automatically auto-loads the best weights from the previous run
    trainer = pl.Trainer()

    predictions = trainer.predict(model, dataloaders=train_loader)

    # # or call with pretrained model
    # model = MyLightningModule.load_from_checkpoint(PATH)
    # trainer = Trainer()
    # predictions = trainer.predict(model, dataloaders=test_dataloader)
    