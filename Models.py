from keras import Input, Model
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, Concatenate
from keras.regularizers import l1, l2
from keras.optimizers import Adam


class CopyCatModel():
    def __init__(self, num_of_fonts, num_of_chars, leaky_relu_alpha, regularization, lr=1):
        reg = l2(regularization)

        fonts = Input((num_of_fonts,))
        emb = Dense(40, kernel_regularizer=reg, name='font_embedding')(fonts)
        chars = Input((num_of_chars,))
        X = Concatenate(name='font_char')([emb, chars])
        X = Dense(1024, kernel_regularizer=reg)(X)
        X = LeakyReLU(leaky_relu_alpha)(X)
        X = Dense(1024, kernel_regularizer=reg)(X)
        X = LeakyReLU(leaky_relu_alpha)(X)
        X = Dense(1024, kernel_regularizer=reg)(X)
        X = LeakyReLU(leaky_relu_alpha)(X)
        X = Dense(1024, kernel_regularizer=reg)(X)
        X = LeakyReLU(leaky_relu_alpha)(X)
        targets = Dense(4096, activation="sigmoid", kernel_regularizer=reg)(X)
        self.model = Model(inputs=[fonts, chars], outputs=targets)
        self.optimizer = Adam(lr)

        self.model.compile(optimizer=self.optimizer, loss="mean_absolute_error", metrics=['binary_accuracy'])

    def predict(self, input):
        return self.model.predict(input)

    def train(self, inputs, targets, bs=512, epochs=1, verbose=0):
        return self.model.fit(inputs, targets, batch_size=bs, verbose=verbose, epochs=epochs)

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer

    def save_weights(self, file):
        self.model.save_weights(file)

    def load_weights(self, file):
        self.model.load_weights(file)
