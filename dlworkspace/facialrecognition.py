"""
Here we have the model or models that deal with facial recognition. This class will be more involved as it will
keep track of identified and unidentified persons. Each individual has a unique feature embedding, and this
class will keep a reference database between identities and embeddings.

We will likely be using FaceNet, which is a siamese neural network that can idenitfy people with a single image.
Multiple images are still required for training, but we can simultaneously classify persons whose identities we are
certain of as well as prepare training data. (probably remotely)

credit for the neural network goes to gttps://github.com/harveyslash

"""
import random
import time
import uuid
import os

import PIL
import cv2
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

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')


def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show(block=False)


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show(block=False)


class Face(object):
    def __init__(self, image, bounds):
        """
        We initalize a face object with these parameters because this object is only used to track in-environment faces.

        :param image: image is the extracted face photo in rgb color and put in a list, with subsequent images being
            used for training and such for identification. This will populate to different levels depending on
            whether or not the person is already in the face database.
        :param bounds: bounds is a tuple with (x,y,w,h)
        :param eyes: the coordinates of the eyes in [(x,y),(x,y),...(if they somehow have more...ew)]
        """
        self.images = [image]
        self.bounds = bounds
        self.old_bounds = None
        self.name = ""  # Set this later

    def add_image(self, image):
        self.images = self.images[-4:] + image


class Faces(object):
    def __init__(self):
        self.model = load_model()
        self.detected_faces = []
        self.identified_faces = {}
        self.faces = {}
        with open(SiameseConfig.vec_path, 'r') as f:
            r = f.readlines()
            for line in r:
                line = line.split("|")
                self.faces[line[0]] = np.array([float(x) for x in line[1].split(",")])

    def save_model(self):
        save_model(self.model)

    def save_collected(self):
        for person in self.identified_faces:
            for image in self.identified_faces[person].images:
                image.save(os.join("data", "siamesetraining", person,
                                   len(os.listdir(os.join("data", "siamesetraining", person + ".bmp")))))

    def save_identities(self):
        with open(SiameseConfig.vec_path, 'w') as f:
            lines = ["{}|{}\n".format(name, ",".join([x for x in self.faces[name]])) for name in self.faces]
            f.writelines(lines)

    def detect_in_current_frame(self, image):
        """
        Returns faces
        :param image:
        :return:
        """
        self.detected_faces = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            self.detected_faces.append(Face(image[x:x + w, y:y + h], (x, y, w, h)))

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 1)

        return image

    def track_faces(self):
        """
        After detecting the faces in a frame, we need to match them with existing faces or add them to the
        database. As this is resource intensive we don't run this real time.
        :return:
        """
        for face in self.detected_faces:
            name = self.identify_face_short(face)[0]
            if name is None:
                name = self.identify_face_long(face)[0]
            if name in self.identified_faces:
                self.identified_faces[name].add_image(face.image)
                self.identified_faces[name].old_bounds = self.identified_faces[name].bounds
                self.identified_faces[name].bounds = face.bounds
                del face
            else:
                face.name = name
                self.identified_faces[name] = face

    def get_diff(self, vec1, vec2):
        return F.pairwise_distance(vec1, vec2)

    def identify_face_short(self, face):
        """
        Uses a bruite force evaluation that attempts to match the encoding vector with either a previously identified
        face or create a new identity. This searches through currently detected faces
        :param image:
        :return:
        """
        img = cv2.cvtColor(face.images[0], cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        transform = transforms.Compose([transforms.Resize(SiameseConfig.imsize, SiameseConfig.imsize),
                                        transforms.ToTensor()])

        img = im_pil.convert("L")
        img = PIL.ImageOps.invert(img)
        input = transform(img)

        output = self.model(input)

        candidates = []
        for person in self.identified_faces:
            person = self.identified_faces[person]
            diff = sum([self.get_diff(image, output) for image in person.images]) / float(len(person.images))
            if diff < 1.0:
                candidates.append((diff, person.name))

        candidates.sort()

        if len(candidates) is 0:
            return None

        return [self.faces[candidate[1]] for candidate in candidates]

    def identify_face_long(self, face):
        """
        Uses a bruite force evaluation that attempts to match the encoding vector with either a previously identified
        face or create a new identity. This goes through the entire face database so it may take a long time.
        :param image:
        :return:
        """
        img = cv2.cvtColor(face.images[0], cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        transform = transforms.Compose([transforms.Resize(SiameseConfig.imsize, SiameseConfig.imsize),
                                        transforms.ToTensor()])

        img = im_pil.convert("L")
        img = PIL.ImageOps.invert(img)
        input = transform(img)

        output = self.model(input)

        candidates = []
        for person in self.faces:
            diff = self.get_diff(self.faces[person], output)
            if diff < 1.0:
                candidates.append((diff, person))

        candidates.sort()

        if len(candidates) is 0:
            newname = "UID: " + str(uuid.uuid4())
            self.faces[newname] = output
            im_pil.save(os.join("data", "siamesetraining", newname, "0.bmp"))
            return [newname]

        im_pil.save(os.join("data", "siamesetraining", candidates[0][1],
                            len(os.listdir(os.join("data", "siamesetraining", candidates[0][1] + ".bmp")))))

        return [self.faces[candidate[1]] for candidate in candidates]


# From here on I lifted the entire thing because I couldn't bother with coming up something original

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
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
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

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


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

            output1, output2 = net(img0, img1)
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
    x0, _, _ = next(dataiter)
    if torch.cuda.is_available():
        net = net.cuda()
    for i in range(10):
        _, x1, label2 = next(dataiter)
        concatenated = torch.cat((x0, x1), 0)

        if torch.cuda.is_available():
            output1, output2 = net(Variable(x0).cuda(), Variable(x1).cuda())
        else:
            output1, output2 = net(Variable(x0), Variable(x1))

        print(output1, output2)
        euclidean_distance = F.pairwise_distance(output1, output2)
        print(euclidean_distance)
        imshow(torchvision.utils.make_grid(concatenated),
               'Dissimilarity: {:.2f}'.format(euclidean_distance.cpu().data.numpy()[0][0]))
        time.sleep(.5)
        plt.close()


def save_model(net):
    torch.save(net.state_dict(), SiameseConfig.model_path)


def load_model():
    net = SiameseNetwork()
    net.load_state_dict(torch.load(SiameseConfig.model_path))
    return net
