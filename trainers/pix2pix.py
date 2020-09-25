# Pix2Pix 256 Trainer
import torch
import numpy as np
import os
from models.network import Network
from contextlib import suppress
# Use PatchGAN Discriminator and CycleGAN Generator
from models.discriminators import PatchGANDiscriminator as Discriminator
from models.generators import CycleGANGenerator as Generator
from losses.vgg import VGGLoss

class Pix2PixTrainer(object):
    def __init__(self, model_name, log_dir='logs', epoch=0, use_latest=False,
            lambda_pixel=100, glr=0.0002, dlr=0.0002, beta=0.5, in_channels=3):
        self.epoch=epoch
        self.lambda_pixel = lambda_pixel
        self.log_dir = log_dir
        # Create checkpoint paths
        self.model_name = model_name
        self.log_dir, self.img_dir = self.make_log_dir()

        # Create the model objects
        self.netG = Generator(in_channels=in_channels)
        self.netD = Discriminator(in_channels=in_channels)

        # loss functions
        self.gan_loss = torch.nn.MSELoss()
        self.pixelwise_loss = torch.nn.L1Loss()
        self.perceptual_loss = VGGLoss()

        # put the models in the GPU
        self.netG.cuda()
        self.netD.cuda()
        self.gan_loss.cuda()
        self.pixelwise_loss.cuda()
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

        # Optimizers and LR schedulers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=glr, betas=(beta, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=dlr, betas=(beta, 0.999))


    def step(self, inputs):
        """ Execute a training step
        """
        source, target = inputs
        batch_size = target.size()[0]
        # Create labels to check against
        label_dims = [batch_size, 1, 16, 16] # for 256 x 256
        fake_labels = torch.zeros(label_dims)
        real_labels = torch.ones(label_dims)

        # Train the Generator
        self.optimizer_G.zero_grad()
        # Pass the source image to the Generator
        generated = self.netG(source.cuda())
        adv_generated = self.netD(generated, source.cuda())
        # Try to trick the discriminator
        loss_g_adv = self.gan_loss(adv_generated, real_labels.cuda())

        # Pixelwise loss
        #loss_pixel = self.pixelwise_loss(generated, target.cuda())

        # Perceptual loss
        loss_perception = self.perceptual_loss(generated, target.cuda())

        # Calculate weighted errors
        #loss_g = self.lambda_pixel * loss_pixel + loss_g_adv + loss_perception
        #loss_g = self.lambda_pixel * loss_pixel + loss_g_adv
        loss_g = loss_g_adv + loss_perception

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
                    #('pixel', loss_pixel.mean().item()),
                    ('percept', loss_perception.mean().item()),
                    ('D(G(x))', loss_d_fake.mean().item()),
                    ('D(x)', loss_d_real.mean().item()),
                    ]
        return ret_loss, generated.detach()


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



