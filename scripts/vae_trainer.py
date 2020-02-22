"""
@author Feiyang Cai
@email feiyang.cai@vanderbilt.edu
@create date 2020-02-21 22:00:47
@modify date 2020-02-21 22:00:47
@desc This is the trainer for the VAE for CARLA AEBS, which is used as an out-of-distribution detector
"""

from scripts.data_loader import CarlaAEBSDataset
from torch.utils.data import DataLoader
import torch
from scripts.network import VAE
import torch.optim as optim
import os
import time
import logging

logger_path = "./log"
if not os.path.exists(logger_path):
    os.makedirs(logger_path)
logging.basicConfig(level=logging.INFO, filename=os.path.join(logger_path, 'vae_epoch_350_Feb_21.log'))

class VAETrainer():
    def __init__(self, data_path, epoch):
        self.dataset = CarlaAEBSDataset(data_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch = epoch
    
    def fit(self):
        self.model = VAE()
        self.model = self.model.to(self.device)
        data_loader = DataLoader(self.dataset, batch_size=64, shuffle=True, num_workers=8)
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.5e-6, amsgrad=False) 
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(self.epoch*0.7)], gamma=0.1)
        self.model.train()
        for epoch in range(self.epoch):
            loss_epoch = 0.0
            reconstruction_loss_epoch = 0.0
            kl_loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()

                x, mu, logvar = self.model(inputs)
                reconstruction_loss = torch.mean((x - inputs)**2, dim=tuple(range(1, x.dim())))
                kl_loss = 1 + logvar - (mu).pow(2) - logvar.exp()
                kl_loss = torch.mean(kl_loss, axis=-1) * -0.5
                loss = reconstruction_loss + kl_loss
                reconstruction_loss_mean = torch.mean(reconstruction_loss)
                kl_loss_mean = torch.mean(kl_loss)
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                reconstruction_loss_epoch += reconstruction_loss_mean.item()
                kl_loss_epoch += kl_loss_mean.item()
                n_batches += 1
            
            scheduler.step()
            epoch_train_time = time.time() - epoch_start_time
            logging.info('Epoch {}/{}\t Time: {:.3f}\t Total Loss: {:.3f}\t\
                        Reconstruction Loss {:.3f}\t KL Loss {:.3f}'\
                        .format(
                            epoch+1, self.epoch, epoch_train_time, loss_epoch/n_batches, reconstruction_loss_epoch/n_batches, \
                            kl_loss_epoch/n_batches))
            print('Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'.format(epoch+1, self.epoch, epoch_train_time, loss_epoch/n_batches))
        return self.model
    
    def save_model(self, path = "./model"):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.model.state_dict(), os.path.join(path, "vae.pt"))