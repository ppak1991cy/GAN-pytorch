import os

import math
import numpy as np
import time
from contextlib import contextmanager

import torch
from torch import autograd
import torchvision
from tensorboardX import SummaryWriter

from utils import chose_device, Reporter
from load_data import GANImageDataset, DataIterator
from net.discriminator import Discriminator
from net.generator import Generator

# --------Parameter--------
device = chose_device()

# Dataset
BATCH_SIZE = 8
dataset_path = "api"

# Iteration
ITERS = 200000
CRITIC_ITERS = 5
GENER_ITERS = 1

# Save dir
SAVE_PATH = "save"
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
MODEL_PATH = os.path.join(SAVE_PATH, "model")
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
IMAGES_PATH = os.path.join(SAVE_PATH, "images")
if not os.path.exists(IMAGES_PATH):
    os.mkdir(IMAGES_PATH)


@contextmanager
def timer(msg):
    start = time.time()
    yield
    end = time.time()
    print(msg + ". Time cost: %.2f" % (end - start))


def gen_rand_noise():
    noise = torch.rand(BATCH_SIZE, 128)
    noise = noise.to(device)
    return noise


def calc_gradient_penalty(net_d, real_data, fake_data):
    """ Refer to 'https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py'.
    """
    lamb = 10
    side_len = int(math.sqrt(int(real_data.nelement() / BATCH_SIZE) / 3))
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, 3 * side_len * side_len).contiguous()
    alpha = alpha.view(BATCH_SIZE, 3, side_len, side_len)
    alpha = alpha.to(device)

    fake_data = fake_data.view(BATCH_SIZE, 3, side_len, side_len)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = net_d(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lamb
    return gradient_penalty


def train():

    # Essential train elements
    image_dataset = GANImageDataset(dataset_path)
    image_dataiter = DataIterator(dataset=image_dataset, batch_size=8, shuffle=True)
    net_d = Discriminator().to(device)
    net_g = Generator().to(device)

    lr = 1e-4
    optimizer_g = torch.optim.Adam(net_d.parameters(), lr=lr, betas=(0, 0.9))
    optimizer_d = torch.optim.Adam(net_g.parameters(), lr=lr, betas=(0, 0.9))
    one = torch.FloatTensor([1]).to(device)
    mone = one * -1

    # Record items
    keys_d = ["fake", "real", "gp", "d_loss"]
    reporter_d = Reporter(keys_d, t="discriminator", interval=1, file_path=None)
    keys_g = ["g_loss"]
    reporter_g = Reporter(keys_g, t="generator", interval=1, file_path=None)
    writer = SummaryWriter()

    fixed_noise = gen_rand_noise()
    with torch.no_grad():
        fixed_noisev = fixed_noise

    # Training process
    for iteration in range(ITERS):

        # 1. Update netD(discriminator)
        for p in net_d.parameters():  # reset requires_grad
            p.requires_grad_(True)  # they are set to False below in netG update
        for i in range(CRITIC_ITERS):
            with timer("Critic iter: " + str(iteration) + "/" + str(i)):
                net_d.zero_grad()
                noise = gen_rand_noise()
                with torch.no_grad():
                    noisev = noise
                fake_data = net_g(noisev).detach()
                real_data = next(image_dataiter)

                # Discriminator loss define
                d_fake = net_d(fake_data).mean()  # Expection of discriminating fake data
                d_real = net_d(real_data).mean()  # Expection of discriminating real data
                gradient_penalty = calc_gradient_penalty(net_d, real_data, fake_data)
                d_loss = d_fake - d_real + gradient_penalty
                # Update parameters
                d_loss.backward(one)
                optimizer_d.step()

            # Record "fake", "real", "gp", "d_loss"
            record_d = [d_fake.data, d_real.data, gradient_penalty.data, d_loss.data]
            record_d = [float(r) for r in record_d]
            reporter_d.recive(record_d)
            if i == CRITIC_ITERS - 1:
                reporter_d.report(iteration)
                writer.add_scalar("data/d_loss", d_loss, iteration)
                writer.add_scalar("data/gradient_penalty", gradient_penalty, iteration)

        # 2. Update netG(generator)
        for p in net_d.parameters():
            p.requires_grad_(False)
        for i in range(GENER_ITERS):
            with timer("Generator iters: " + str(iteration) + "/" + str(i)):
                net_g.zero_grad()
                noise = gen_rand_noise()
                noise.requires_grad_(True)
                fake_data = net_g(noise)

                # Generator loss define
                g_loss = net_d(fake_data).mean()
                # Update parameters
                g_loss.backward(mone)
                optimizer_g.step()

            # Record "g_loss"
            record_g = [g_loss.data]
            record_g = [float(r) for r in record_g]
            reporter_g.recive(record_g)
            if i == GENER_ITERS - 1:
                reporter_g.report(iteration)
                writer.add_scalar("data/d_loss", - g_loss, iteration)

        # Save intermediate models and images
        if iteration % 200 == 199:
            v = (iteration + 1) / 200
            gen_images = net_g(fixed_noisev) * 0.5 + 0.5  # Scale from -1 ~ 1 to 0 ~ 1
            torchvision.utils.save_image(gen_images, os.path.join(IMAGES_PATH, "samples_{}.png".format(v)),
                                         nrow=BATCH_SIZE, padding=2)
            grid_images = torchvision.utils.make_grid(gen_images, nrow=8, padding=2)
            writer.add_image('images', grid_images, iteration)
            torch.save(net_d, os.path.join(MODEL_PATH, "discriminator_{}.pkg".format(v)))
            torch.save(net_g, os.path.join(MODEL_PATH, "generator_{}.pkg".format(v)))


if __name__ == "__main__":
    train()
