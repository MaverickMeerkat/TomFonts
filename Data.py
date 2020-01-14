from keras.utils import HDF5Matrix, to_categorical
import h5py
import numpy as np


def preprocess(data):
    hf = h5py.File('./Data/targets.hdf5', 'w')
    # fonts = []
    # chars = []
    targets = []
    # hf.create_dataset('fonts', (3499466, 56443))
    # hf.create_dataset('chars', (3499466, 62))
    hf.create_dataset('targets', (3499466, 4096))

    s = c = 0
    try:
        for font in range(len(data)):
            for char in range(len(data[0])):
                # fonts.append(font)
                # chars.append(char)
                target = np.array(data[font][char]).reshape(4096,)
                targets.append(target)
                c += 1
            if c % 62000 == 0:
                # fonts = font_one_hot(fonts)
                # chars = one_hot(chars)
                # hf['fonts'][s:c] = fonts
                # hf['chars'][s:c] = chars
                hf['targets'][s:c] = targets
                # fonts = []
                # chars = []
                targets = []
                s = c
                print("batch completed " + str(c))
    finally:
        hf.close()


def load_file(file, name):
    return HDF5Matrix(file, name)


def one_hot(k):
    return to_categorical(k)

def font_one_hot(fonts):
    s = np.zeros((len(fonts), 56443))
    s[np.arange(len(fonts)), fonts] = 1
    return s