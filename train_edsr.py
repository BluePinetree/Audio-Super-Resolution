import argparse
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.losses import MeanAbsoluteError
from data import VCTK
from model.edsr import edsr, edsr2
from train import EDSRTrainer
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay, ExponentialDecay

def main():
    batch = 8
    epoch = 20
    loss = MeanAbsoluteError()
    learning_rate = PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 5e-5])
    res_blocks = [15, 15, 15, 15, 15, 9, 9, 9, 9, 9, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3]
    checkpoint_dir = './ckpt/edsr'

    # 데이터 가져오기
    ds_train = VCTK(subset='train').dataset()
    ds_valid = VCTK(subset='valid').dataset()

    # 모델 빌딩
    edsr_model = edsr2(scale=4, res_blocks=res_blocks, res_block_scaling=0.7)

    # 훈련
    edsr_trainer = EDSRTrainer(model=edsr_model, loss=loss, learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)
    edsr_trainer.train(train_dataset=ds_train, valid_dataset=ds_valid, batch=batch, epoch=epoch)

    edsr_model.save_weights(f'./weights/EDSR_16000_{len(res_blocks)}res_{batch}batch_{epoch}epochs_tanh_entropy_glorot_uniform.h5')

if __name__ == '__main__':
    main()