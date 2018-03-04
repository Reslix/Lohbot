"""
Here we have the model or models that deal with facial recognition. This class will be more involved as it will
keep track of identified and unidentified persons. Each individual has a unique feature embedding, and this
class will keep a reference database between identities and embeddings.

We will likely be using FaceNet, which is a siamese neural network that can idenitfy people with a single image.
Multiple images are still required for training, but we can simultaneously classify persons whose identities we are
certain of as well as prepare training data. (probably remotely)

credit for the neural network goes to gttps://github.com/harveyslash

"""
import os
import uuid

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

from jetson.siamesenetwork import SiameseConfig, load_model, save_model

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')


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
        self.vector = None

    def add_image(self, image):
        self.images = self.images[-4:] + [image]


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
                self.faces[line[0]] = Variable(torch.from_numpy(
                    np.array([float(x) for x in line[1].split(",")], dtype=np.float32)) \
                                               .view(1, 8))
        if torch.cuda.is_available():
            for face in self.faces:
                self.faces[face] = self.faces[face].cuda()

    def save_model(self):
        save_model(self.model)

    def save_collected(self):
        for person in self.identified_faces:
            for image in self.identified_faces[person].images:
                if not os.path.exists(os.path.join("data", "siamesetraining", person)):
                    os.mkdir(os.path.join("data", "siamesetraining", person))
                image.save(os.path.join("data", "siamesetraining", person, str(
                    len(os.listdir(os.path.join("data", "siamesetraining", person)))) + ".bmp"))

    def save_identities(self):
        with open(SiameseConfig.vec_path, 'w') as f:
            combined = self.faces.copy()
            for name in self.identified_faces:
                combined[name] = self.identified_faces[name].vector
            lines = ["{}|{}\n".format(name, ",".join(
                [str(combined[name].data[0][x]) for x in range(combined[name].size()[1])])) for name in combined]
            f.writelines(lines)

    def detect_in_current_frame(self, image):
        """
        Returns faces
        :param image:
        :return:
        """
        self.detected_faces = []

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.5, 5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            x = x - w // 6
            y = y - h // 4
            w = w + w // 3
            h = h + h // 2

            if w >= SiameseConfig.imsize and h >= SiameseConfig.imsize:
                self.detected_faces.append(Face(image.copy()[y:y + h, x:x + w], (x, y, w, h)))

        for (x, y, w, h) in faces:
            x = x - w // 6
            y = y - h // 4
            w = w + w // 3
            h = h + h // 2

            if w >= SiameseConfig.imsize and h >= SiameseConfig.imsize:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

        return image

    def single_face_capture(self, name):
        for face in self.detected_faces:
            im_pil = Image.fromarray(face.images[0])
            if not os.path.exists(os.path.join("data", "siamesetraining", name)):
                os.mkdir(os.path.join("data", "siamesetraining", name))
            im_pil.save(os.path.join("data", "siamesetraining", name, str(
                len(os.listdir(os.path.join("data", "siamesetraining", name)))) + ".bmp"))

    def track_faces(self):
        """
        After detecting the faces in a frame, we need to match them with existing faces or add them to the
        database. As this is resource intensive we don't run this real time.
        :return:
        """
        for face in self.detected_faces:
            name = self.identify_face_short(face)
            if name is None:
                name = self.identify_face_long(face)
            face.name = name
            if name in self.identified_faces:
                self.identified_faces[name].add_image(face.images[0])
                self.identified_faces[name].old_bounds = self.identified_faces[name].bounds
                self.identified_faces[name].bounds = face.bounds
            else:
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
        print("face_short")
        print(face.name)
        im_pil = Image.fromarray(face.images[0])
        transform = transforms.Compose([transforms.Resize((SiameseConfig.imsize, SiameseConfig.imsize)),
                                        transforms.ToTensor()])

        img = im_pil.convert("L")
        input = transform(img).view(1, 1, SiameseConfig.imsize, SiameseConfig.imsize)

        if torch.cuda.is_available():
            output = self.model(Variable(input).cuda())
        else:
            output = self.model(Variable(input))

        candidates = []
        for person in self.identified_faces:
            person = self.identified_faces[person]
            diff = self.get_diff(person.vector, output)
            if diff.data[0][0] < 1.0:
                candidates.append((diff.data[0][0], person.name))
                person.vector += (person.vector - torch.mean(torch.cat((output, person.vector), dim=0), dim=0)) * .01

        candidates.sort()

        if len(candidates) is 0:
            return None

        return candidates[0][1]

    def identify_face_long(self, face):
        """
        Uses a bruite force evaluation that attempts to match the encoding vector with either a previously identified
        face or create a new identity. This goes through the entire face database so it may take a long time.
        :param image:
        :return:
        """
        print("face_long")
        im_pil = Image.fromarray(face.images[0])
        transform = transforms.Compose([transforms.Resize((SiameseConfig.imsize, SiameseConfig.imsize)),
                                        transforms.ToTensor()])

        img = im_pil.convert("L")
        input = transform(img).view(1, 1, SiameseConfig.imsize, SiameseConfig.imsize)

        if torch.cuda.is_available():
            output = self.model(Variable(input).cuda())
        else:
            output = self.model(Variable(input))
        candidates = []
        for person in self.faces:
            diff = self.get_diff(self.faces[person], output)
            if diff.data[0][0] < 1.0:
                candidates.append((diff.data[0][0], person))

        candidates.sort()

        if len(candidates) is 0:
            newname = str(uuid.uuid4().int % 2000)
            face.name = newname
            face.vector = output
            if not os.path.exists(os.path.join("data", "siameseraw", newname)):
                os.mkdir(os.path.join("data", "siameseraw", newname))
            im_pil.save(os.path.join("data", "siameseraw", newname, "0.bmp"))
            face.name = newname
            face.vector = output
            self.save_identities()
            return newname

        if not os.path.exists(os.path.join("data", "siameseraw", candidates[0][1])):
            os.mkdir(os.path.join("data", "siameseraw", candidates[0][1]))

        im_pil.save(os.path.join("data", "siameseraw", candidates[0][1], str(
            len(os.listdir(os.path.join("data", "siameseraw", candidates[0][1])))) + ".bmp"))

        face.name = candidates[0][1]
        face.vector = self.faces[candidates[0][1]]

        return candidates[0][1]
