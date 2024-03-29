from model.DQ.dq import *
import torch
import torch.nn as nn
import pytorch_lightning as pl
from dataloader import Dataload
from model.util import *
import os
from torch.utils.data import DataLoader

device = 'cuda:1'

class DFGAN(pl.LightningModule):

    def __init__(self, in_channels, out_channels, learning_rate=0.0002, lambda_recon=100, display_step=10, lambda_gp=10, lambda_r1=10, middle_channel =  [64,128,256,512]):
        super().__init__()
        self.save_hyperparameters()
        self.display_step = display_step
        self.nstack = 2
        self.generator = DQAE(
            in_channel = in_channels,
            channel = 256,
            n_res_block = 3,
            n_res_channel = 256,
            n_coder_blocks = 2,
            embed_dim = 64,
            n_codebooks = 5,
            stride = 2,
            decay = 0.99,
            loss_name = "mse",
            vq_type = "dq",
            beta = 0.25,
            n_hier = middle_channel,
            n_logistic_mix = 10,
        )
        self.critic = Critic(out_channels)
        self.generator.apply(self._weights_init)
        self.critic.apply(self._weights_init)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
        self.optimizer_C = torch.optim.Adam(self.critic.parameters(), lr=learning_rate, betas=(0.5, 0.9))
        self.lambda_recon = lambda_recon
        self.lambda_gp = lambda_gp
        self.lambda_r1 = lambda_r1
        self.recon_criterion = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.generator_losses, self.critic_losses  = [],[]
        self.save_path = 'D:'
    
    def _weights_init(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    def configure_optimizers(self):
        return [self.optimizer_C, self.optimizer_G]
        
    def generator_step(self, real_images, conditioned_images):
        # WGAN has only a reconstruction loss
        self.optimizer_G.zero_grad()
        fake_images = self.generator(conditioned_images)
      
        recon_loss = 0
        beta = (self.nstack + 1) * self.nstack / 2
        for i in range(self.nstack):
            alpha = (i+1)/self.nstack
            # label = alpha * real_images + ( 1 - alpha ) * conditioned_images
            recon_loss += ( alpha / beta ) * self.recon_criterion(fake_images[i], real_images)

        #recon_loss = self.recon_criterion(fake_images, real_images)
        recon_loss.backward()
        self.optimizer_G.step()
        
        # Keep track of the average generator loss
        self.generator_losses += [recon_loss.item()/self.nstack]
        
        
    def critic_step(self, real_images, conditioned_images):
        self.optimizer_C.zero_grad()
        fake_images = self.generator(conditioned_images)
        fake_images = fake_images[-1]
        fake_logits = self.critic(fake_images)
        real_logits = self.critic(real_images)
        
        # Compute the loss for the critic
        loss_C = real_logits.mean() - fake_logits.mean()

        #Compute the gradient penalty
        alpha = torch.rand(real_images.size(0), 1, 1, 1, requires_grad=True)
        alpha = alpha.to(device)
        interpolated = (alpha * real_images + (1 - alpha) * fake_images.detach()).requires_grad_(True)
        
        interpolated_logits = self.critic(interpolated)
        
        grad_outputs = torch.ones_like(interpolated_logits, dtype=torch.float32, requires_grad=True)
        gradients = torch.autograd.grad(outputs=interpolated_logits, inputs=interpolated, grad_outputs=grad_outputs,create_graph=True, retain_graph=True)[0]

        
        gradients = gradients.view(len(gradients), -1)
        gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        loss_C += self.lambda_gp * gradients_penalty
        
        #Compute the R1 regularization loss
        r1_reg = gradients.pow(2).sum(1).mean()
        loss_C += self.lambda_r1 * r1_reg

        # Backpropagation
        loss_C.backward()
        self.optimizer_C.step()
        self.critic_losses += [loss_C.item()]
        
    def training_step(self, batch, batch_idx, optimizer_idx):
        condition, real = batch
        if optimizer_idx == 0:
            self.critic_step(real, condition)
        elif optimizer_idx == 1:
            self.generator_step(real, condition)
        gen_mean = sum(self.generator_losses[-self.display_step:]) / self.display_step
        crit_mean = sum(self.critic_losses[-self.display_step:]) / self.display_step
        if self.current_epoch%self.display_step==0 and batch_idx==0 and optimizer_idx==1:
            fake = self.generator(condition)
            fake = fake[-1].detach()
            torch.save(self.generator.state_dict(), self.save_path +"/ResUnet_"+ str(self.current_epoch) +".pt")
            torch.save(self.critic.state_dict(), self.save_path +"/PatchGAN_"+ str(self.current_epoch) +".pt")
            print(f"Epoch {self.current_epoch} : Generator loss: {gen_mean}, Critic loss: {crit_mean}")
            display_progress(condition[0], real[0], fake[0], self.current_epoch, (20,15),True, self.save_path)
        
if __name__ == '__main__':

    batch_size = 16
    image_size = 224
    train_path = r'/home/data/lijl/DATA/Color/dark/'
    label_path = r'/home/data/lijl/DATA/Color/val/'
    #All_dataloader = Dataload(r'H:\DATASET\COLORDATA\train\train_frame', r'H:\DATASET\COLORDATA\train_gt\train_gt_frame')
    All_dataloader = Dataload(train_path,label_path, image_shape = (image_size, image_size))
    train_size = int(len(All_dataloader.photo_set) * 0.8)
    validate_size = len(All_dataloader.photo_set) - train_size

    train_dataset, validate_dataset = torch.utils.data.random_split(All_dataloader
                                                                    , [train_size, validate_size])
    print("训练集大小： {} 测试集大小： {} , ".format(train_size, validate_size))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    validate_loader = DataLoader(
        dataset=validate_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    cwgan = DFGAN(in_channels = 3, out_channels = 3 ,learning_rate=2e-4, lambda_recon=100, display_step=5, middle_channel= [32,64,128,256])
    cwgan.save_path = "./save/dqgan/"
    if not os.path.exists(cwgan.save_path):
        os.mkdir(cwgan.save_path)
    trainer = pl.Trainer(max_epochs=50, gpus = [1])
    #cwgan.generator.load_state_dict(torch.load('./save/ResUnet_90.pt'))
    #cwgan.critic.load_state_dict(torch.load('./save/PatchGAN_90.pt'))
    trainer.fit(cwgan, train_loader)
