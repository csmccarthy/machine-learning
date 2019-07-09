from matplotlib import pyplot as plt
import numpy as np
from math import floor, ceil
import pickle
import cv2
import random
import gzip
from mnistHelper import mnistHelper

class DataAug:
    """ Stores and augments MNIST data in order to train a more robust neural network """

    percent = None
    augfactor = None #[1, 2, 2, 2, 1, 1, 0, 0] works pretty well
    transforms = None
    data = None
    labels = None
    img_size = 28
    savefolder = "C:\\Users\\tmccarthy\\Documents\\pythonprojects\\machine-learning\\MNIST-augmented\\"
    
    def __init__(self, multipliers):
        """ Loads MNIST, stores augfactor (res, rot, x, y, c, sp, ga, un) """

        self.augfactor = multipliers
        self.data, self.labels = mnistHelper.load_data(train=True, blanks = True)
        self.data = self.data.reshape([-1, self.img_size, self.img_size])
        self.data = list(self.data)
        self.labels = list(self.labels)
        self.percent = 100/len(self.data)
        self.transforms = [self.resize, self.rotate, self.xtrans, self.ytrans,
                           self.contrast, self.saltnpepper, self.gaussian, self.uniform]


    def augment(self):
        """ Runs all augmentation functions on data """

        for factor in self.augfactor:
            if factor > 1:
                newlbls = []
                for i in range(len(self.labels)):
                    lbl = self.labels.pop(0)
                    for j in range(factor):
                        newlbls.append(lbl)
                    self.labels.extend(newlbls)
                    newlbls = []

        for i in range(len(self.transforms)):
            if self.augfactor[i] != 0:
                self.transforms[i]()
                    
        print('Done!')

    
    def contrast(self):
        """ Varies the image intensity levels and clips to be within normal bounds """

        augs = []
        for i in range(len(self.data)):
            if i % 100 == 0: print(f"Contrast variation is {int(i*self.percent)}% done")
            img = self.data.pop(0)
            for j in range(self.augfactor[4]):
                contrast = (random.random()*0.3)+0.2
                shifted = img-contrast
                clipped = np.clip(shifted, 0, 1)
                augs.append(floored)
            self.data.extend(augs)
            augs = []


    def xtrans(self, shiftrange=2):
        """ Shifts the image horizontally, fills empty space with zeros """

        augs = []
        for i in range(len(self.data)):
            if i % 100 == 0: print(f"X variation is {int(i*self.percent)}% done")
            img = self.data.pop(0)
            for j in range(self.augfactor[2]):
                x_dis = random.randint(-shiftrange, shiftrange)
                fill = np.zeros([self.img_size, abs(x_dis)], dtype=np.float64)
                if x_dis < 0:
                    shifted = img[:, :x_dis]
                    shifted = np.concatenate((fill, shifted), axis=1)
                elif x_dis > 0:
                    shifted = img[:, x_dis:]
                    shifted = np.concatenate((shifted, fill), axis=1)
                else:
                    shifted = img
                augs.append(shifted)
            self.data.extend(augs)
            augs = []


    def ytrans(self, shiftrange=2):
        """ Shifts the image vertically, fills empty space with zeros """

        augs = []
        for i in range(len(self.data)):
            if i % 100 == 0: print(f"Y variation is {int(i*self.percent)}% done")
            img = self.data.pop(0)
            for j in range(self.augfactor[3]):
                y_dis = random.randint(-shiftrange, shiftrange)
                fill = np.zeros([abs(y_dis), self.img_size], dtype=np.float64)
                if y_dis < 0:
                    shifted = img[:y_dis, :]
                    shifted = np.concatenate((fill, shifted), axis=0)
                elif y_dis > 0:
                    shifted = img[y_dis:, :]
                    shifted = np.concatenate((shifted, fill), axis=0)
                else:
                    shifted = img
                augs.append(shifted)
            self.data.extend(augs)
            augs = []


    def saltnpepper(self, noise_ratio=0.025):
        """ Chooses random pixels and sets them to either 1 or 0 """
        
        augs = []
        squares = self.img_size*self.img_size
        noise = int(squares*noise_ratio)
        idx = np.concatenate((np.ones(noise), np.zeros(squares-noise)))
        for i in range(len(self.data)):
            if i % 100 == 0: print(f"Salt and pepper noise variation is {int(i*self.percent)}% done")
            img = self.data.pop(0)
            for j in range(self.augfactor[5]):
                np.random.shuffle(idx)
                choice = np.reshape(idx, (self.img_size,self.img_size))
                temp = np.where(choice == 1, 0, img)

                np.random.shuffle(idx)
                choice = np.reshape(idx, (self.img_size,self.img_size))
                temp = np.where(choice == 1, 1, temp)
                augs.append(temp)
            self.data.extend(augs)
            augs = []

    def gaussian(self, mu = 0, sigma = 0.2):
        """ Adds gaussian noise """
        
        augs = []
        for i in range(len(self.data)):
            if i % 100 == 0: print(f"Gaussian noise variation is {int(i*self.percent)}% done")
            img = self.data.pop(0)
            for j in range(self.augfactor[6]):
                noise = np.random.normal(mu, sigma, img.shape)
                noisy_img = np.clip(noise+img, 0, 1)
                augs.append(noisy_img)
            self.data.extend(augs)
            augs = []

    def uniform(self, start = -0.5, stop = 0.5):
        """ Adds uniform noise """
        
        augs = []
        for i in range(len(self.data)):
            if i % 100 == 0: print(f"Uniform noise variation is {int(i*self.percent)}% done")
            img = self.data.pop(0)
            for j in range(self.augfactor[7]):
                scale = stop-start
                noise = (np.random.rand(*img.shape)*scale)+start
                noisy_img = np.clip(noise+img, 0, 1)
                augs.append(noisy_img)
            self.data.extend(augs)
            augs = []
    
            
    def rotate(self, maxangle=15):
        """ Randomly rotates image about its center """
        
        augs = []
        for i in range(len(self.data)):
            if i % 100 == 0: print(f"Rotational variation is {int(i*self.percent)}% done")
            img = self.data.pop(0)
            for i in range(self.augfactor[1]):
                angle = (random.random()*2*maxangle)-maxangle
                parent = np.zeros([40, 40], dtype=np.float64)
                parent[6:34, 6:34] = img
                rot = cv2.getRotationMatrix2D((20,20), angle, 1)
                parent = cv2.warpAffine(parent, rot, (40,40))
                temp = parent[6:34, 6:34]
                augs.append(temp)
            self.data.extend(augs)
            augs = []


    def resize(self):
        """ Randomly scales or shrinks the image """
        
        augs = []
        for i in range(len(self.data)):
            if i % 100 == 0: print(f"Size variation is {int(i*self.percent)}% done")
            img = self.data.pop(0)
            for i in range(self.augfactor[0]):
                ratio = (random.random()*0.2)+0.8
                temp = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
                padval = (self.img_size-(temp.shape[0]))/2
                pad = (floor(padval),ceil(padval))
                temp = np.pad(temp, (pad,pad), 'constant', constant_values=0)
                augs.append(temp)
            self.data.extend(augs)
            augs = []


    def save(self, name):
        """ Pickles augmented images to a folder for later use """

        with open(self.savefolder + "aug_data_" + name + ".pkl", 'wb') as f:
            pickle.dump(self.data, f)
        with open(self.savefolder + "aug_labels_" + name + ".pkl", 'wb') as f:
            pickle.dump(self.labels, f)


    def load(self, name):
        """ Unpickles and returns augmented images for training """
        
        with open(self.savefolder + "aug_data_" + name + ".pkl", 'rb') as f:
            data = pickle.load(f)
        with open(self.savefolder + "aug_labels_" + name + ".pkl", 'rb') as f:
            labels = pickle.load(f)
            

    def reset(self):
        """ Resets MNIST data back to its unaugmented form """

        self.data, self.labels = mnistHelper.load_data(train=True, blanks = True)
        self.data = self.data.reshape([-1, self.img_size, self.img_size])
        self.data = list(self.data)
        self.labels = list(self.labels)

    def setAugfactor(self):
        """ Sets a new array of augmentation factors to be used """

        for i, transform in enumerate(self.transforms):
            newfactor = int(input("Augmentation factor for " + transform.__name__ + ": "))
            self.augfactor[i] = newfactor
        

    def preview(self, num):
        """ Randomly chooses n images to preview """

        picks = np.random.choice(len(self.data), num)
        for pick in picks:
            plt.imshow(self.data[pick])
            plt.show()

    

    
