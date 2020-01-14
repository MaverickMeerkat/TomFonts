import keras
import numpy as np
# from keras.utils import HDF5Matrix, to_categorical


class DataGenerator(keras.utils.Sequence):
    # Generates data for Keras
    def __init__(self, hf, batch_size=62, shuffle=True):
        self.hf = hf
        self.data_size = len(self.hf) * len(self.hf[0])
        self.indexes = np.arange(self.data_size)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return self.data_size // self.batch_size

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        fonts = np.zeros((self.batch_size, len(self.hf)))
        chars = np.zeros((self.batch_size, len(self.hf[0])))

        f_i = indexes // len(self.hf[0])
        c_i = indexes % len(self.hf[0])
        fonts[np.arange(len(f_i)), f_i] = 1
        chars[np.arange(len(c_i)), c_i] = 1

        targets = self.hf[]
        # targets = np.zeros((self.batch_size, self.hf.shape[2]*self.hf.shape[3]))
        # for i in range(self.batch_size):
        #     targets[i] = self.hf[f_i[i]][c_i[i]].reshape(self.hf.shape[2] * self.hf.shape[3],)

        return [fonts, chars], targets
