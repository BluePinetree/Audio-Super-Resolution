import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE
from time import sleep
from scipy import signal, misc
from scipy.io import wavfile
from utils import play_sound, fourier_transform

class VCTK:
    def __init__(self,
                 scale=4,
                 subset='train',
                 audio_dir='./VCTK-Corpus/wav48',
                 cache_dir='./VCTK-Corpus/caches',
                 val_dir='p376',
                 max_fs=8000):

        # 크기 검사
        scales = [2,4]
        if scale in scales:
            self.scale=scale
        else:
            ValueError(f'scale must be in {scales}')

        # 훈련데이터, 검증데이터
        if subset == 'train':
            self.data_ids = [os.path.join(audio_dir, folder) for folder in os.listdir(audio_dir) if folder != val_dir]
        elif subset == 'valid':
            self.data_ids = [os.path.join(audio_dir, val_dir)]
        else:
            ValueError('subset must be \'train\' or \'valid\'')

        self.subset = subset
        self.audio_dir = audio_dir
        self.cache_dir = cache_dir
        self.max_fs = max_fs
        self.ls_fs = int(self.max_fs/self.scale)
        os.makedirs(cache_dir, exist_ok=True)

    def hr_dataset(self):
        # 파일 경로들 불러오기 후 하나의 리스트에 담아오기
        files = self._get_filepath()
        print(files[0])
        # 파일을 읽어 Dataset 객체로 가져오기
        ds = self._audio_dataset(files)
        # wav파일을 디코딩하고 data만 가져온다.
        ds = ds.map(lambda x : tf.audio.decode_wav(x)[0], num_parallel_calls=AUTOTUNE)
        # 미리 설정해놓은 최고 주파수로 줄인다.
        ds = ds.map(lambda x : self._preprocess_hr_data(x), num_parallel_calls=AUTOTUNE)
        return ds

    def lr_dataset(self):
        files = self._get_filepath()
        ds = self._audio_dataset(files)
        ds = ds.map(lambda x : tf.audio.decode_wav(x)[0], num_parallel_calls=AUTOTUNE)
        ds = ds.map(self._preprocess_lr_data, num_parallel_calls=AUTOTUNE)
        return ds

    def _preprocess_hr_data(self, x):
        x = tf.reshape(x, (-1,))
        # x = x.numpy().ravel()
        x = x[:len(x) - (len(x) % self.max_fs)]

        scale_factor = int(len(x) / self.max_fs)
        if scale_factor > 13:
            half = scale_factor // 2
            scaled_data = tf.py_function(signal.decimate, [x, half], tf.float32)
            scaled_data = tf.py_function(signal.decimate, [scaled_data, 2], tf.float32)
            # scaled_data = signal.decimate(x, half)
            # scaled_data = signal.decimate(scaled_data, 2)
        else:
            scaled_data = tf.py_function(signal.decimate, [x, scale_factor], tf.float32)
            # scaled_data = signal.decimate(x, scale_factor)

        return scaled_data

    def _preprocess_lr_data(self, x):
        x = x.numpy().ravel()
        x = x[:len(x) - (len(x) % self.lr_fs)]

        scale_factor = len(x) / self.lr_fs
        if scale_factor > 13:
            half = scale_factor // 2
            scaled_data = signal.decimate(x, half)
            scaled_data = signal.decimate(scaled_data, 2)
        else:
            scaled_data = signal.decimate(x, scale_factor)

        return scaled_data

    def _get_filepath(self):
        print('Configuring filepaths...')
        en_path = []
        for dir in self.data_ids:
            data = [os.path.join(dir, file) for file in os.listdir(dir) if file[0] != '.']
            print(f'There are {len(data)} files in {dir}...')
            en_path.extend(data)
        print('Finished!')
        print(f'Total num of filepaths .. {len(en_path)}')
        return en_path

    @staticmethod
    def _audio_dataset(files):
        # 데이터셋 클래스 정의
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(tf.io.read_file)
        return ds