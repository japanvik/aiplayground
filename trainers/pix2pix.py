# Pix2Pix 256 Trainer
import torch
import numpy as np
import os
from models.network import init_weights
# Use PatchGAN Discriminator and CycleGAN Generator
from models.discriminators import PatchGANDiscriminator as Discriminator
from models.generators import CycleGANGenerator as Generator

class Pix2PixTrainer(object):
    def __init__(self, model_name, log_dir='logs', epoch=0, lambda_pixel=100, lr=0.0002, beta=0.5):
        self.epoch=epoch
        self.lambda_pixel = lambda_pixel
        # Create checkpoint paths
        self.model_name = model_name
	self.log_dir, self.img_dir = self.make_log_dir()

        # Create the model objects
        self.netG = Generator()
        self.netD = Discriminator()

        # loss functions
        self.gan_loss = torch.nn.MSELoss()
        self.pixelwise_loss = torch.nn.L1Loss()

        # put the models in the GPU
        self.netG.cuda()
        self.netD.cuda()
        self.gan_loss.cuda()
        self.pixelwise_loss.cuda()

        # Initialize weights
        if self.epoch == 0:
            self.netG.apply(init_weights)
            self.netD.apply(init_weights)
        else:
            # load weights from a checkpoint
            self.netG.load(identifier=self.model_name, iteration=self.epoch, path=self.log_dir)
            self.netD.load(identifier=self.model_name, iteration=self.epoch, path=self.log_dir)

        # Optimizers and LR schedulers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(beta, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(beta, 0.999))


    def step(self, inputs):
        """ Execute a training step
        """
        source = inputs['source'].cuda()
        target = inputs['target'].cuda()
        batch_size = target.size()[0]
        # Train the Generator
        self.optimizer_G.zero_grad()
        # Pass the source image to the Generator
        generated = self.netG(source)
        # Calculate errors
        loss_g = self.pixelwise_loss(generated, target)

        loss_g.backward()
        self.optimizer_G.step()

        # Train the Discriminator
        self.optimizer_D.zero_grad()
        # Check if the generated image can be correctly detected
        label_dims = [batch_size, 1, 16, 16] # for 256 x 256
        adv_fake = self.netD(source, generated.detatch())
        fake_labels = torch.zeros(label_dims)
        loss_d_fake = self.gan_loss(adv_fake, fake_labels.cuda())

        adv_real = self.netD(source, target)
        real_labels = torch.ones(label_dims)
        loss_d_real = self.gan_loss(adv_real, real_labels.cuda())

        loss_d = loss_d_fake + loss_d_real
        loss_d.backward()
        self.optimizer_G.step()

        # Values to keep track of
        L_G = loss_g.mean().item()
        L_D = loss_d.mean().item()
        ##########################################
        ret_loss = [('lossD', loss_d.mean().item()),
                    ('lossG', loss_g.mean().item()),
                    ('D(G(x))', loss_d_fake.mean().item()),
                    ('D(x)', loss_d_real.mean().item()),
                    ]
        return ret_loss, generated.detatch()


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

