import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import datasets, transforms
from tqdm import tqdm

from ex02_model import Unet
from ex02_diffusion import Diffusion, linear_beta_schedule, cosine_beta_schedule
from torchvision.utils import save_image


def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to diffuse images')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--timesteps', type=int, default=100, help='number of timesteps for diffusion model (default: 100)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    # parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--run_name', type=str, default="DDPM")
    parser.add_argument('--dry_run', action='store_true', default=False, help='quickly check a single pass')
    return parser.parse_args()


def train(model, trainloader, optimizer, diffusor, epoch, device, args):
    batch_size = args.batch_size
    timesteps = args.timesteps

    pbar = tqdm(trainloader)
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Algorithm 1 line 3: sample t uniformly for every example in the batch
        t = torch.randint(0, timesteps, (len(images),), device=device).long()
        loss = diffusor.p_losses(model, images, t, labels, loss_type="l2")

        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(images), len(trainloader.dataset),
                100. * step / len(trainloader), loss.item()))
        if args.dry_run:
            break


def test(model, testloader, diffusor, device, args):
    # TODO: Implement - adapt code and method signature as needed
    timesteps = 5 # just 5 timestamps for test
    loss = 0
    with torch.no_grad():
        pbar = tqdm(testloader)
        for step, (images, labels) in enumerate(pbar):

            images = images.to(device)
            labels = labels.to(device)
            # Algorithm 1 line 3: sample t uniformly for every example in the batch
            t = torch.randint(0, timesteps, (len(images),), device=device).long()
            loss += diffusor.p_losses(model, images, t, labels, loss_type="l2").item()


    loss /= len(testloader)
    print("Test Loss is {:.4f}".format(loss))
    return loss


def sample_and_save_images(n_images, image_size, diffusor, model, device, store_path, reverse_transform):
    # TODO: Implement - adapt code and method signature as needed
    batch_size = 8
    num_samples = n_images // batch_size
    images = []
    for _ in range(num_samples):
        images.append(diffusor.sample(model, image_size, batch_size=batch_size, channels=3))

    imgs = images[0][-1] # one batch for visualisation
    nrows = (batch_size // 3) + 1
    ncols = 3
    fig, ax = plt.subplots(nrows, ncols)
    for i in range(len(imgs)):
        img = imgs[i]
        img = reverse_transform(img)

        row = i // 3
        col = i % 3
        ax[row, col].imshow(img)
    plt.savefig(os.path.join(store_path, "1.png"))


def visualize_test(model, testloader, diffusor, device, store_path, reverse_transform, random_index):
    random_idx = random_index
    t_list = [0, 5, 10, 50, 99]

    with torch.no_grad():
        pbar = tqdm(testloader)
        for step, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

        plot_forward_diffusion(
            [get_noisy_image(images[random_idx], diffusor, torch.tensor([t]), reverse_transform) for t in t_list],
            store_path,
            index=random_idx
        )


def get_noisy_image(x_start, diffusor, t, reverse_transform):
    # add noise
    x_noisy = diffusor.q_sample(x_start, t=t)
    # turn back into PIL image
    noisy_image = reverse_transform(x_noisy.squeeze())
    return noisy_image


def plot_forward_diffusion(imgs, store_path, index, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        imgs = [imgs]
    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(figsize=(200, 200), nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    print("SAVING THE IMAGE!")
    plt.savefig(os.path.join(store_path, f"forward_diffusion-{index}.png"))


def visualize_sampling(n_samples, model, diffusor, image_size, store_path, reverse_transform):
    random_index = 5
    batch_size = 8
    channels = 3
    timesteps = 100
    images = []

    for _ in range(n_samples):
        samples = diffusor.sample(model, image_size, batch_size=batch_size, channels=channels)
        images.append(samples)

    fig = plt.figure()
    ims = []
    for i in range(timesteps):
        image_tensor = samples[i][random_index]
        print("image tensor shape: ", image_tensor.shape)
        image_array = reverse_transform(image_tensor)
        # Plot the image using Matplotlib
        im = plt.imshow(image_array, cmap="gray", animated=True)
        ims.append([im])

    animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    animate.save(os.path.join(store_path, "diffusion.gif"))
    plt.show()


def run(args):
    timesteps = args.timesteps
    image_size = 32  # TODO (2.5): Adapt to new dataset
    channels = 3
    num_classes = 10
    epochs = args.epochs
    batch_size = args.batch_size
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"

    model = Unet(dim=image_size,
                channels=channels,
                dim_mults=(1, 2, 4,),
                class_free_guidance=True,
                num_classes=num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    my_linear_scheduler = lambda x: linear_beta_schedule(0.0001, 0.02, x)
    my_cosine_scheduler = lambda x: cosine_beta_schedule(x)
    diffusor = Diffusion(timesteps, my_linear_scheduler, image_size, device)

    # define image transformations (e.g. using torchvision)
    transform = Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),    # turn into torch Tensor of shape CHW, divide by 255
        transforms.Lambda(lambda t: (t * 2) - 1)   # scale data to [-1, 1] to aid diffusion process
    ])
    reverse_transform = Compose([
        Lambda(lambda t: (t.clamp(-1, 1) + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    dataset = datasets.CIFAR10('/proj/aimi-adl/CIFAR10/', download=True, train=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    # Download and load the test data
    testset = datasets.CIFAR10('/proj/aimi-adl/CIFAR10/', download=True, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=int(batch_size/2), shuffle=True)
    save_path = "images_linear_CFG_10_epochs/"  # TODO: Adapt to your needs
    n_images = 8
    for epoch in range(epochs):
        train(model, trainloader, optimizer, diffusor, epoch, device, args)
        #test(model, valloader, diffusor, device, args)
        if (epoch + 1) % 2 == 0:
            sample_and_save_images(n_images, image_size, diffusor, model, device, save_path, reverse_transform)

    # test(model, testloader, diffusor, device, args)

    sample_and_save_images(n_images, image_size, diffusor, model, device, save_path, reverse_transform)
    torch.save(model.state_dict(), os.path.join("./models", args.run_name, f"linear_cfg_10_ckpt.pt"))


if __name__ == '__main__':
    args = parse_args()
    # TODO (2.2): Add visualization capabilities
    run(args)
