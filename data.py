import os
import gc
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa
from tensorflow.python.data.experimental import AUTOTUNE
from time import sleep
from scipy import signal, misc
from tqdm import tqdm
from utils import play_sound, fourier_transform, VCTKSRSequence, normalize

class VCTK:
    def __init__(self,
                 scale=4,
                 subset='train',
                 audio_dir='./VCTK-Corpus/wav48',
                 cache_dir='./VCTK-Corpus/caches',
                 val_dir='p376',
                 max_fs=16000):

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
        self.data_dir = './data'
        self.max_fs = max_fs
        self.ls_fs = int(self.max_fs/self.scale)
        os.makedirs(cache_dir, exist_ok=True)

    def dataset(self):
        # 파일 경로들 불러오기 후 하나의 리스트에 담아오기기
        files = self._get_filepath()
        print(files[0])
        # 파일을 읽어 Dataset 객체로 가져오기
        ds_hr, ds_lr = self._audio_dataset(files, self.data_dir)

        # HR, LR 데이터를 튜플로 묶어 random_cropping
        ds = tf.data.Dataset.zip((ds_lr, ds_hr))
        if self.subset == 'train':
            ds = ds.map(lambda lr, hr : random_cropping(lr, hr, scale=self.scale, hr_crop_size=self.max_fs//2), num_parallel_calls=AUTOTUNE)

        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def _get_filepath(self):
        print('Configuring filepaths...')
        en_path = []
        for dir in self.data_ids:
            data = [os.path.join(dir, file) for file in os.listdir(dir) if file[0] != '.' and file[-3:] == 'wav']
            print(f'There are {len(data)} files in {dir}...')
            en_path.extend(data)
        print('Finished!')
        print(f'Total num of filepaths .. {len(en_path)}')
        return en_path

    def _preprocessed_file(self):
        return f'VCTK_{self.subset}_fs{str(self.max_fs)}_X{str(self.scale)}.h5'

    def _get_cache_file(self):
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        file = f'VCTK_{self.subset}_fs{str(self.max_fs)}_X{str(self.scale)}.cache'
        return os.path.join(self.cache_dir, file)

    def _get_cache_index(self):
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        file = f'VCTK_{self.subset}_fs{str(self.max_fs)}_X{str(self.scale)}.index'
        return os.path.join(self.cache_dir, file)

    # 다운샘플링 함수
    def _decimate_audio(self, data, scale):
        if scale > 13:
            half = scale / 2
            scaled_data = signal.decimate(data, half)
            scaled_data = signal.decimate(scaled_data, 2)
        else:
            scaled_data = signal.decimate(data, scale)

        return scaled_data

    def _make_dataset(self, files, data_path):
        hr_list = []
        lr_list = []

        print(f'Preprocessing {self.subset} data...')
        # for i, file in enumerate(tqdm(files)):
        for file in tqdm(files):
            data, rate = librosa.load(file, None)
            # 다운샘플링
            hr_factor = int(rate / self.max_fs)
            hr_data = self._decimate_audio(data, hr_factor)
            # Low Resolution 다운샘플링
            hr_data = hr_data[:len(hr_data) - (len(hr_data) % self.scale)]  # 다시 나눠지게 잘라준다
            lr_data = self._decimate_audio(hr_data, self.scale)

            # # Check
            # if ((i+1) % 100) == 0:
            #     print(len(data) / self.max_fs)
            #     print(hr_factor)
            #     print('Original :', len(data) / rate)
            #     print('HR_length :', len(hr_data) / self.max_fs)
            #     print('LR_length :', len(lr_data) / self.ls_fs)

            # 전처리한 데이터들 저장
            hr_list.append(hr_data)
            lr_list.append(lr_data)
        print(f'\nPreprocessing finished! | HR : {len(hr_list)}, LR : {len(lr_list)}')

        # 파일로 저장
        h5_file = h5py.File(data_path, 'w')
        dt = h5py.vlen_dtype(np.dtype('float32'))
        hr_dataset = h5_file.create_dataset('HR_data', shape=(len(hr_list),), dtype=dt)     # 가변길이 데이터셋 준비
        lr_dataset = h5_file.create_dataset('LR_data', shape=(len(lr_list),), dtype=dt)
        for i, d in enumerate(tqdm(hr_list)) :   hr_dataset[i] = d       # 데이터 삽입
        for i, d in enumerate(tqdm(lr_list)) :   lr_dataset[i] = d
        h5_file.close()
        print(f'Sucessfully saved at {data_path}!')
        gc.collect()

    def _audio_dataset(self, files, data_path):
        file_path = os.path.join(data_path, self._preprocessed_file())
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        if not os.path.exists(file_path):
            self._make_dataset(files, file_path)

        h5_file = h5py.File(file_path, 'r')
        print(h5_file.keys())
        hr_data = h5_file['HR_data'][...]
        lr_data = h5_file['LR_data'][...]
        assert hr_data.shape == lr_data.shape
        print(hr_data.shape, lr_data.shape, hr_data.dtype)

        print(type(hr_data))
        print(type(lr_data))

        hr_datagen = VCTKSRSequence(hr_data)
        lr_datagen = VCTKSRSequence(lr_data)

        print(callable(hr_datagen), callable(lr_datagen))

        ds_hr = tf.data.Dataset.from_generator(hr_datagen, tf.float32, tf.TensorShape([None]))
        ds_lr = tf.data.Dataset.from_generator(lr_datagen, tf.float32, tf.TensorShape([None]))

        return ds_hr, ds_lr


def random_cropping(lr_data, hr_data, hr_crop_size=4000, scale=4):
    lr_crop_size = hr_crop_size//scale
    lr_shape = tf.shape(lr_data)[0]

    lr_st_idx = tf.random.uniform(shape=(), maxval=lr_shape - lr_crop_size + 1, dtype=tf.dtypes.int32)
    hr_st_idx = lr_st_idx * scale

    lr_cropped = lr_data[lr_st_idx:lr_st_idx+lr_crop_size]
    hr_cropped = hr_data[hr_st_idx:hr_st_idx+hr_crop_size]

    return lr_cropped, hr_cropped