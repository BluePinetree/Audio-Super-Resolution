import os
import gc
import tensorflow as tf
import numpy as np
import librosa

from scipy.io import wavfile
from scipy.signal import decimate
from model.common import *
from model.edsr import edsr, edsr2
from utils import normalize, denormalize
from tqdm import tqdm

def main():
    data_dir = './VCTK-Corpus/wav48'
    person = 'p374'
    path = os.path.join(data_dir, person)
    res_blocks = [15, 15, 15, 15, 15, 9, 9, 9, 9, 9, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3]

    files = [os.path.join(path,file) for file in os.listdir(path) if file[0] != '.' and file[-3:] == 'wav']
    # idx = np.random.choice(range(len(files)), 1)
    idx = 3

    evaluate_val(files, person, res_blocks, True)


def evaluate_val(files, person, res_blocks, save_files=False):
    # get model
    edsr_model = edsr(scale=4, res_blocks=res_blocks, res_block_scaling=0.5)
    edsr_model.load_weights('./weights/EDSR_16000_20res_8batch_10epochs_TV+abserr.h5')

    for idx, file in enumerate(tqdm(files)):
        data, rate = librosa.load(file, None)
        # data = normalize(data)
        # print(f'Original rate : {rate}')
        data = data[:len(data) - (len(data) % 3)]
        data = np.asarray(decimate(data, 3), dtype=np.float32)
        hr_rate = int(rate / 3)
        # print(f'High Resolution rate : {rate}')
        # downsampling
        data = data[:len(data) - (len(data) % 4)]
        lr = np.asarray(decimate(data, 4), dtype=np.float32)
        lr_rate = int(hr_rate / 4)
        # print(f'Low Resolution rate : {lr_rate}')
        # reshape
        lr = np.asarray(lr).reshape((-1, lr.shape[0], 1))
        hr = np.asarray(data).reshape((-1, data.shape[0], 1))
        # resolve
        sr = resolve(edsr_model, lr)
        sr = sr.numpy()
        sr = np.reshape(sr, (sr.shape[1],))

        if save_files:
            lr = lr.reshape((lr.shape[1],))
            librosa.output.write_wav(f'./result/{person}_{idx + 1}_LR_data_{lr_rate}.wav', lr, lr_rate)
            librosa.output.write_wav(f'./result/{person}_{idx + 1}_HR_data_{hr_rate}.wav', data, hr_rate)
            librosa.output.write_wav(f'./result/{person}_{idx + 1}_SR_data_{hr_rate}.wav', sr, hr_rate)

        gc.collect()

if __name__ == '__main__':
    main()