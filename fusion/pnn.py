
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from sklearn.preprocessing import LabelEncoder
import datetime
from tensorflow.python.profiler import profiler_v2 as profile

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)

def cross_layer(x0, xl):
    embed_dim = 27 * 64
    w = tf.Variable(tf.random.truncated_normal(shape=(embed_dim,), stddev=0.01))
    b = tf.Variable(tf.zeros(shape=(embed_dim,)))
    x1_T = tf.reshape(xl, [-1, 1, embed_dim])
    x_lw = tf.tensordot(x1_T, w, axes=1)
    cross = x0 * x_lw
    return cross + b + xl

def build_cross_layer(x0, num_layer=3):
    x1 = x0
    for i in range(num_layer):
        x1 = cross_layer(x0, x1)
    return x1

with tf.device("GPU:0"):

    input_layer = tf.keras.Input(shape=(27 * 64,))
    embed_inputs = tf.random.normal(shape=(1024, 27 * 64), mean=0.0, stddev=1.0)
    label = tf.random.uniform(shape=(1024, 1), minval=0, maxval=2, dtype=tf.int32)

    cross_layer_output = build_cross_layer(input_layer, 3)
    fc_layer = Dropout(0.5)(Dense(128, activation='relu')(input_layer))
    fc_layer = Dropout(0.3)(Dense(128, activation='relu')(fc_layer))
    fc_layer_output = Dropout(0.1)(Dense(128, activation='relu')(fc_layer))
    stack_layer = Concatenate()([cross_layer_output, fc_layer_output])
    output_layer = Dense(1, activation='sigmoid', use_bias=True)(stack_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['binary_crossentropy', tf.keras.metrics.AUC(name='auc')])

    tf.profiler.experimental.start(log_dir)
    model.fit(embed_inputs, label, epochs=5, batch_size=1024)
    tf.profiler.experimental.stop()
