from model.common import evaluate, resolve, normalize

import time
import tensorflow as tf
import numpy as np
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

class Trainer:
    def __init__(self,
                 model,
                 loss,
                 learning_rate,
                 checkpoint_dir='./ckpt/edsr'):
        # 실행 시간 저장용
        self.now = None
        self.loss = loss
        # 체크포인트와, 체크포인트 매니저 생성
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              snr=tf.Variable(-1),
                                              optimizer=Adam(learning_rate),
                                              model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=3)

        self.restore()

    @property
    def model(self):
        return self.checkpoint.model

    def train(self, train_dataset, valid_dataset, data_length, batch=8, epoch=50, save_best_only=True):
        # 손실함수 평균을 내기위해
        loss_mean = Mean()

        # 체크포인트 가져오기
        ckpt = self.checkpoint
        ckpt_mgr = self.checkpoint_manager

        # 훈련 데이터셋 배치 적용
        train_dataset = train_dataset.batch(batch)
        train_dataset = train_dataset.repeat()

        # 1에폭 연산속도를 재기위해 시간 카운트 시작
        self.now = time.perf_counter()

        steps_per_epoch = int(data_length/batch)
        i = steps_per_epoch // ckpt.step.numpy()

        print(f'Total steps : {steps_per_epoch*epoch}, current steps : {ckpt.step.numpy()}')

        for lr, hr in train_dataset.take(steps_per_epoch*epoch - ckpt.step.numpy()):
            # step에 1씩 더해준다.
            ckpt.step.assign_add(1)
            # 훈련
            loss = self.train_step(lr, hr)
            loss_mean(loss)

            if (ckpt.step % 1000) == 0:
                print(f'Step:{ckpt.step.numpy()}, Loss : {loss_mean.result()}')

            if (ckpt.step % steps_per_epoch) == 0:
                loss_value = loss_mean.result()
                loss_mean.reset_states()

                # 검증 데이터셋으로 평가
                snr_value = self.evaluate(valid_dataset)

                # 실행 시간 평가
                duration = time.perf_counter() - self.now
                print(f'{i+1}/{epoch} : loss = {loss_value.numpy():.5f}, SNR = {snr_value.numpy():.5f}, duration = {duration:.3f}s')
                i += 1

                if save_best_only and snr_value.numpy() <= ckpt.snr:
                    self.now = time.perf_counter()
                    continue

                ckpt.snr = snr_value
                ckpt_mgr.save()

                self.now = time.perf_counter()

    @tf.function
    def train_step(self, lr, hr):
        lr = tf.reshape(lr, shape=(-1, lr.shape[1], 1))
        hr = tf.reshape(hr, shape=(-1, hr.shape[1], 1))
        with tf.GradientTape() as tape:
            sr = self.checkpoint.model(lr, training=True)
            loss = -tf.math.log(1 - tf.abs(sr-hr))
            loss = tf.clip_by_value(loss, 0., 5.)
            loss = tf.reduce_mean(loss)
            # loss = self.loss(sr, hr)

        gradient = tape.gradient(loss, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(zip(gradient, self.checkpoint.model.trainable_variables))

        return loss

    def evaluate(self, dataset):
        return evaluate(self.checkpoint.model, dataset)

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')


class EDSRTrainer(Trainer):
    def __init__(self,
                 model,
                 loss=MeanAbsoluteError(),
                 learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5]),
                 checkpoint_dir='./ckpt/edsr'):
        super().__init__(model, loss, learning_rate, checkpoint_dir)

    def train(self, train_dataset, valid_dataset, data_length=43951, batch=8, epoch=50, save_best_only=True):
        super().train(train_dataset, valid_dataset, data_length, batch, epoch, save_best_only)
