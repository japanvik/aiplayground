# Progressive enhancement 256 Trainer
import torch
import numpy as np
import os
from models.network import Network
from contextlib import suppress
# Use PatchGAN Discriminator and CycleGAN Generator
from models.discriminators import PatchGANDiscriminator as Discriminator
from models.generators import CycleGANGenerator as Generator
from losses.vgg import PerceptualLoss

class ProgressiveEnhancement(object):
    def __init__(self, model_name, log_dir='logs', epoch=0, use_latest=False,
            lambda_percept=1, glr=0.0002, dlr=0.0004, beta=0.5,
            in_channels=6, out_channels=3):
        self.epoch=epoch
        self.lambda_percept = lambda_percept
        self.log_dir = log_dir
        # Create checkpoint paths
        self.model_name = model_name
        self.log_dir, self.img_dir = self.make_log_dir()

        # Create the model objects
        self.netFilter = Generator(in_channels=out_channels)
        self.netG = Generator(in_channels=in_channels)
        self.netD = Discriminator(in_channels=out_channels)

        # loss functions
        self.gan_loss = torch.nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()

        # put the models in the GPU
        self.netFilter.cuda()
        self.netG.cuda()
        self.netD.cuda()
        self.gan_loss.cuda()
        self.perceptual_loss.cuda()

        # Initialize weights
        if not use_latest and self.epoch == 0:
            self.netG.apply(Network.init_weights)
            self.netD.apply(Network.init_weights)
        else:
            # load weights from a checkpoint
            lbl = 'latest' if use_latest else epoch
            self.netG.load(identifier=self.model_name, iteration=lbl, path=self.log_dir)
            self.netD.load(identifier=self.model_name, iteration=lbl, path=self.log_dir)

        # Initialize the filter Generator
        self.netFilter.eval()
        self.netFilter.load('fixBlur', '70', './logs/fixBlur') #hardcoded for now

        # Optimizers and LR schedulers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=glr, betas=(beta, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=dlr, betas=(beta, 0.999))


    def step(self, inputs):
        """ Execute a training step
        """
        source, target = inputs

        # Train the Generator
        self.optimizer_G.zero_grad()
        # Pass the source image to the Filter
        with torch.no_grad():
            filtered = self.netFilter(source.cuda())

        img_input = torch.cat((source.cuda(), filtered), 1)
        # Pass the source image to the Generator
        generated = self.netG(img_input)
        # Run it through the Discriminator
        adv_generated = self.netD(generated, source.cuda())

        # Create labels to check against
        label_dims = adv_generated.size()
        fake_labels = torch.zeros(label_dims)
        real_labels = torch.ones(label_dims)

        # Try to trick the discriminator
        loss_g_adv = self.gan_loss(adv_generated, real_labels.cuda())

        # Perceptual loss
        loss_perception = self.perceptual_loss(generated, target.cuda())

        # Calculate weighted errors
        loss_g = loss_g_adv + self.lambda_percept * loss_perception

        loss_g.backward()
        self.optimizer_G.step()

        # Train the Discriminator
        self.optimizer_D.zero_grad()
        # Check if the generated image can be correctly detected

        adv_fake = self.netD(generated.detach(), source.cuda())
        loss_d_fake = self.gan_loss(adv_fake, fake_labels.cuda())

        adv_real = self.netD(target.cuda(), source.cuda())
        loss_d_real = self.gan_loss(adv_real, real_labels.cuda())

        loss_d = loss_d_fake + loss_d_real
        loss_d.backward()
        self.optimizer_D.step()

        # Values to keep track of
        L_G = loss_g.mean().item()
        L_D = loss_d.mean().item()
        ##########################################
        ret_loss = [('lossD', loss_d.mean().item()),
                    ('lossG', loss_g.mean().item()),
                    ('D(G(x))1', loss_g_adv.mean().item()),
                    ('percept', loss_perception.mean().item()),
                    ('D(G(x))', loss_d_fake.mean().item()),
                    ('D(x)', loss_d_real.mean().item()),
                    ]
        return ret_loss, [filtered.detach(), generated.detach()]


    def make_log_dir(self):
        """
        Make checkpoint directories
        """
        log_dir = os.path.join(self.log_dir, self.model_name)
        img_out = os.path.join(log_dir,'checkpoint', 'image')
        with suppress(OSError):
            os.makedirs(log_dir)
            os.makedirs(img_out)
        return log_dir, img_out


    def save(self, label):
        self.netG.save(identifier=self.model_name, iteration=label, path=self.log_dir)
        self.netD.save(identifier=self.model_name, iteration=label, path=self.log_dir)



