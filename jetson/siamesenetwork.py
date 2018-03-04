import random
import time

import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from jetson.show import imshow, show_plot


class SiameseConfig():
    training_dir = "./data/siamesetraining/"
    testing_dir = "./data/siamesetesting/"
    model_path = "./models/siamese.torch"
    vec_path = "./data/kookynsa.txt"
    train_batch_size = 64
    train_number_epochs = 100
    imsize = 96


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Copy-pasted form: https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None, should_invert=False):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            img1_tuple = random.choice(self.imageFolderDataset.imgs)

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * SiameseConfig.imsize * SiameseConfig.imsize, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 8)
        )

    def forward(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output


def train_siamese(net):
    folder_dataset = dset.ImageFolder(root=SiameseConfig.training_dir)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=transforms.Compose(
                                                [transforms.Resize((SiameseConfig.imsize, SiameseConfig.imsize)),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor()
                                                 ])
                                            , should_invert=False)
    train_dataloader = DataLoader(siamese_dataset,
                                  shuffle=True,
                                  num_workers=1,
                                  batch_size=SiameseConfig.train_batch_size)
    if torch.cuda.is_available():
        net = net.cuda()

    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)

    counter = []
    loss_history = []
    iteration_number = 0

    for epoch in range(0, SiameseConfig.train_number_epochs):
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            if torch.cuda.is_available():
                img0, img1, label = Variable(img0).cuda(), Variable(img1).cuda(), Variable(label).cuda()
            else:
                img0, img1, label = Variable(img0), Variable(img1), Variable(label)

            output1, output2 = net(img0), net(img1)
            optimizer.zero_grad()
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            if i % 10 == 0:
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.data[0]))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.data[0])
    show_plot(counter, loss_history)


def test_siamese(net):
    folder_dataset_test = dset.ImageFolder(root=SiameseConfig.testing_dir)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                            transform=transforms.Compose(
                                                [transforms.Resize((SiameseConfig.imsize, SiameseConfig.imsize)),
                                                 transforms.ToTensor()
                                                 ])
                                            , should_invert=False)

    test_dataloader = DataLoader(siamese_dataset, num_workers=1, batch_size=1, shuffle=True)
    dataiter = iter(test_dataloader)

    if torch.cuda.is_available():
        net = net.cuda()
    for j in range(min(5, len(dataiter) // 10)):
        x0, _, _ = next(dataiter)
        for i in range(5):
            _, x1, label2 = next(dataiter)
            concatenated = torch.cat((x0, x1), 0)

            if torch.cuda.is_available():
                output1, output2 = net(Variable(x0).cuda()), net(Variable(x1).cuda())
            else:
                output1, output2 = net(Variable(x0)), net(Variable(x1))

            euclidean_distance = F.pairwise_distance(output1, output2)
            imshow(np.transpose(torchvision.utils.make_grid(concatenated).numpy(), (1, 2, 0)),
                   'Dissimilarity: {:.2f}'.format(euclidean_distance.cpu().data.numpy()[0][0]))
            time.sleep(.5)
            plt.close()


def save_model(net):
    torch.save(net.state_dict(), SiameseConfig.model_path)


def load_model():
    net = SiameseNetwork()
    net.load_state_dict(torch.load(SiameseConfig.model_path))
    if torch.cuda.is_available():
        net = net.cuda()
    return net


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='Deals with Siamese Network training and such')
    # Parse command line arguments
    parser.add_argument("command",
                        metavar="<command>",
                        help="'new', 'train', or 'evaluate'")
    args = parser.parse_args()

    if args.command == "new":
        print("New network created")
        save_model(SiameseNetwork())
    elif args.command == "train":
        net = load_model()
        train_siamese(net)
        save_model(net)
    elif args.command == "evaluate":
        net = load_model()
        test_siamese(net)
