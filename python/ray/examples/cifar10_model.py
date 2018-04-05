#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function

import argparse
import numpy as np
import os

import tensorflow as tf

from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

import ray
from ray.tune import grid_search, run_experiments
from ray.tune import register_trainable
from ray.tune import Trainable
from ray.tune import TrainingResult
from ray.tune.pbt import PopulationBasedTraining


# config = tf.ConfigProto(log_device_placement=True)
# config.gpu_options.per_process_gpu_memory_fraction = 0.25
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))


num_classes = 10


class Cifar10Model(Trainable):

  def _read_data(self):
    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Convert class vectors to binary class matrices.
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')
    x_train /= 255
    x_test = x_test.astype('float32')
    x_test /= 255

    return (x_train, y_train), (x_test, y_test)

  def _build_model(self, input_shape):
    x = Input(shape=(32, 32, 3))
    y = x
    y = Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_initializer='he_normal')(y)
    y = Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_initializer='he_normal')(y)
    y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)

    y = Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_initializer='he_normal')(y)
    y = Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_initializer='he_normal')(y)
    y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)

    y = Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_initializer='he_normal')(y)
    y = Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_initializer='he_normal')(y)
    y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)

    y = Flatten()(y)
    y = Dropout(self.config['dropout'])(y)
    y = Dense(units=10, activation='softmax', kernel_initializer='he_normal')(y)

    model = Model(inputs=x, outputs=y, name='model1')

    return model

  def _setup(self):
    self.train_data, self.test_data = self._read_data()
    print("train data: {}".format(self.train_data[0].shape))
    print("test data: {}".format(self.test_data[0].shape))
    x_train = self.train_data[0]
    model = self._build_model(x_train.shape[1:])
    # initiate RMSprop optimizer
    opt = tf.keras.optimizers.Adadelta()

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    self.model = model

  def _train(self):
    # x_train, y_train = self.train_data
    x_train, y_train = self.test_data
    x_test, y_test = self.test_data

    self.model.fit(x=x_train[:128], y=y_train[:128],
                   epochs=self.config['epochs'],
                   verbose=1,
                   validation_data=None)

    # loss, accuracy
    _, accuracy = self.model.evaluate(x_test[:128], y_test[:128], verbose=0)
    return TrainingResult(
      timesteps_this_iter=10, mean_accuracy=accuracy)

  def _save(self, checkpoint_dir):
    file_path = checkpoint_dir + '/model'
    self.model.save_weights(file_path)
    return file_path

  def _restore(self, path):
    self.model.load_weights(path)

  def _stop(self):
    print("save model")
    # self.model.save(self.logdir)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  args, _ = parser.parse_known_args()

  register_trainable('train_cifar10', Cifar10Model)
  train_spec = {
    'run': 'train_cifar10',
    'stop': {
      # 'mean_accuracy': 0.9,
      'timesteps_total': 60,
    },
    'config': {
      'epochs': 1,
      'batch_size': 64,
      'lr': grid_search([10 ** -3, 10 ** -4, 10 ** -5]),
      'decay': lambda spec: spec.config.lr / 100.0,
      'dropout': grid_search([0.25, 0.5, 0.65]),
    },
    "repeat": 1,
  }

  ray.init()

  pbt = PopulationBasedTraining(
    time_attr="timesteps_total", reward_attr="mean_accuracy",
    perturbation_interval=10,
    hyperparam_mutations={
      'dropout': lambda _: np.random.uniform(0, 1),
    })

  run_experiments(
    {"pbt_cifar10": train_spec}, scheduler=pbt)
