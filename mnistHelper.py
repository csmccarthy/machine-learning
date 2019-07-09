import numpy as np
import gzip

class mnistHelper:
    """ Helper class, used for quickly loading MNIST data and making small adjustments to the datasets """
    
    def load_data(train, blanks = False):
        """ Loads MNIST training data and returns a normalized version with pixel values from 0 to 1 """
        """ Optionally, can append on blank values """

        if (train):
            prefix = "train"
        else:
            prefix = "t10k"
        
        file = gzip.open('C:\\Users\\tmccarthy\\Documents\\pythonprojects\\machine-learning\\MNIST-data\\' + prefix + '-images-idx3-ubyte.gz', 'r')
        file.read(4)
        num_imgs = int.from_bytes(file.read(4), 'big')
        img_size = int.from_bytes(file.read(4), 'big')
        file.read(4)
        buf = file.read(img_size*img_size*num_imgs)
        train_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        train_data = train_data.reshape([num_imgs, img_size, img_size, 1])/255

        file = gzip.open('C:\\Users\\tmccarthy\\Documents\\pythonprojects\\machine-learning\\MNIST-data\\' + prefix + '-labels-idx1-ubyte.gz', 'r')
        file.read(4)
        num_lbls = int.from_bytes(file.read(4), 'big')
        buf = file.read(num_lbls)
        train_labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)

        if (blanks):
            blanks = np.zeros([num_imgs//10, img_size, img_size, 1], dtype=np.float32)
            train_data = np.concatenate((train_data, blanks), axis=0)
            blank_labels = np.ones([6000,], dtype=np.int32)*10
            train_labels = np.concatenate((train_labels, blank_labels), axis=0)

        return train_data, train_labels
