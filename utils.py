import math
import numpy as np
import matplotlib.pyplot as plt
import pygame
import tensorflow as tf


def fourier_transform(signal, rate):
    F = np.fft.fft(signal)
    mag = np.abs(F)
    pha = np.angle(F)
    length = len(signal)
    freq_bin = rate / length
    w = np.arange(0, length).astype('float64')
    w *= freq_bin
    plt.plot(w, mag)
    plt.show()


def play_sound(filename):
    pygame.mixer.init()
    sound = pygame.mixer.Sound(filename)
    tmp = sound.play()
    while tmp.get_busy():
        pygame.time.delay(1)

def normalize(x):
    x = np.cast['float32'](x)
    return x / (2**15)

def denormalize(x):
    return np.cast['int16'](x * (2**15))

# data generator
class VCTKSRSequence(tf.keras.utils.Sequence):

    def __init__(self, data):
        self.data = data
        self.n = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __call__(self, *args, **kwargs):
        return self

    def __iter__(self):
        for item in (self[i] for i in range(len(self))):
            yield item

    def __next__(self):
        if self.n < len(self.data):
            yield self.data[self.n]
            self.n += 1
        else:
            raise StopIteration