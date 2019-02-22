# -*-coding:utf-8-*-
from __future__ import division, print_function, absolute_import

import numpy as np
import h5py
import time
import os
import scipy.io as sio
import tensorflow as tf
from net.DCAE_v2 import DCAE_v2_feature as DCAE_fea
from net.DCAE_v2 import DCAE_v2 as DCAE
from net.DCAE_v2 import DCAE_LR
from evaluate.PSNR import psnr
from sklearn.model_selection import StratifiedKFold
# import preprocess.preprocess as pre
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import argparse
import keras
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model

model_name = model_name+'_{}.model'.format(epoch)
feature_model = DCAE_fea()
feature_model.load_weights(self.model_name, by_name=True)
