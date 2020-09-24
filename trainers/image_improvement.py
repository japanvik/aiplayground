# image_improvement 256 Trainer
import torch
import numpy as np
import os
from models.network import Network
from contextlib import suppress
# use CycleGAN Generator
from models.generators import CycleGANGenerator as Generator
from losses.vgg import PerceptualLoss

class ImageImprovementTrainer(object):
    def __init__(self, model_name, log_dir='logs', epoch=0, use_latest=False,
            lambda_percept=1, lr=0.0002, beta=0.5, in_channels=3):
        self.epoch=epoch
        self.lambda_percept = lambda_percept
        self.log_dir = log_dir
        # Create checkpoint paths
        self.model_name = model_name
        self.log_dir, self.img_dir = self.make_log_dir()

        # Create the model objects
        self.netG = Generator(in_channels=in_channels)

        # loss functions
        self.perceptual_loss = PerceptualLoss()

        # put the models in the GPU
        self.netG.cuda()
        self.perceptual_loss.cuda()

        # Initialize weights
        if not use_latest and self.epoch == 0:
            self.netG.apply(Network.init_weights)
        else:
            # load weights from a checkpoint
            lbl = 'latest' if use_latest else epoch
            self.netG.load(identifier=self.model_name, iteration=lbl, path=self.log_dir)

        # Optimizers and LR schedulers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(beta, 0.999))


    def step(self, inputs):
        """ Execute a training step
        """
        source, target = inputs
        batch_size = target.size()[0]

        # Train the Generator
        self.optimizer_G.zero_grad()
        # Pass the source image to the Generator
        generated = self.netG(source.cuda())


        # Perceptual loss
        loss_perception = self.perceptual_loss(generated, target.cuda())

        loss_g = self.lambda_percept * loss_perception

        loss_g.backward()
        self.optimizer_G.step()

        # Values to keep track of
        ##########################################
        ret_loss = [('lossG', loss_g.mean().item()),
                    ('percept', loss_perception.mean().item()),
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



