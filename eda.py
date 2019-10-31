import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal, misc
import pygame
import tensorflow as tf
from pygame import mixer
from time import sleep
from data import VCTK

print(tf.version.VERSION)

def play_sound(filename):
    pygame.mixer.init()
    sound = pygame.mixer.Sound(filename)
    tmp = sound.play()
    while tmp.get_busy():
        pygame.time.delay(1)

def get_wav_name(dir, file):
    return os.path.join(dir, file)

def main(data_dir, p_name):
    index = 1   # sample index
    person = os.path.join(data_dir, p_name)
    # # Get Audio files
    filenames = [fname for fname in os.listdir(person)]
    # # Play sample files
    # # play_sound(get_wav_name(person, filenames[index]))
    # # Read file
    # rate, data = wavfile.read(get_wav_name(person, filenames[index]))
    # print(data.min())
    # text = os.path.join(person,filenames[index])
    # file = tf.io.read_file(text)
    # data, rate = tf.audio.decode_wav(file)
    #
    # print(type(data))

    """ 다운샘플링한거 주파수영역에서 보기 """
    # F = np.fft.fft(data.numpy().ravel())
    # mag = np.abs(F)
    # freq = rate.numpy()/len(mag)
    # w = np.arange(0, len(mag)) * freq
    # plt.plot(w,mag)
    # plt.show()
    # print(mag.min(), mag.max())

    # #TODO: scale 조정(decode_wav로 했다면 필수)
    # data = data.numpy()
    # print(((2**15)*data).astype('int16').min(), rate)

    # downsampled_data = signal.decimate(data.numpy().ravel(), 12)
    # downsampled_data = signal.decimate(downsampled_data, 2)
    # downsampled_data = downsampled_data * (2**15)
    # print(len(downsampled_data) / 2000)

    # print(downsampled_data.reshape((-1,)).shape)

    # wavfile.write('test1.wav', 2000, downsampled_data.astype('int16'))

    # a = [[1,2,3,4], [2,3,4], [3,5,6], [4,5,6], [5,1]]
    # a = np.array(a)
    # # print(type(a[0]))
    # file = h5py.File('./data/test.h5', 'w')
    # dt = h5py.vlen_dtype(np.dtype('int16'))
    # dataset = file.create_dataset('test', shape=(5,), dtype=dt)
    # for i, d in enumerate(a):
    #     dataset[i] = d
    # file.close()

    # TODO:hdf5파일 원소들의 길이가 가변인 파일 저장시키기
    # file = h5py.File('./data/test.h5', 'r')
    # data = file['test'][...]
    # d_gen = data_generator(data)
    # for file in d_gen:
    #     print(file.dtype)

    ds_hr, ds_lr = VCTK().dataset()

    i = 1
    for hr, lr in zip(ds_hr.repeat(1), ds_lr.repeat(1)):
        if i%1000 == 0:
            print(i)
        i += 1
        # print(hr.shape, lr.shape)
    print(i)

def data_generator(data):
    i = 0
    n = len(data)
    while i<data:
        yield data[i]
        i += 1

if __name__ == '__main__':
    data_dir = './VCTK-Corpus/wav48'
    p_name = 'p226'
    main(data_dir, p_name)