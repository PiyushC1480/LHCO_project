import tensorflow as tf
import torch
from tensorflow.keras.layers import Layer ,BatchNormalization, Lambda
import torch.nn as nn
import torch.optim as optim
import uproot
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import os, ast
import torch.nn.functional as F
# -------------------------- PARTICLE CLOUD MDOEL -------------------------------
# A shape is (N, P_A, C), B shape is (N, P_B, C)
# D shape is (N, P_A, P_B)

def batch_distance_matrix_general(A, B):
    def compute_dist(inputs):
        A, B = inputs
        r_A = tf.reduce_sum(A * A, axis=2, keepdims=True)
        r_B = tf.reduce_sum(B * B, axis=2, keepdims=True)
        m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
        D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
        return D
    return Lambda(compute_dist)([A, B])

def knn(num_points, k, topk_indices, features):
    def gather_fn(inputs):
        topk_indices, features = inputs
        batch_size = tf.shape(features)[0]
        batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, num_points, k, 1))
        indices = tf.concat([batch_indices, tf.expand_dims(topk_indices, axis=3)], axis=3)
        return tf.gather_nd(features, indices)

    def output_shape_fn(input_shapes):
        # input_shapes = [(batch, num_points, k), (batch, num_points, c)]
        topk_shape, feat_shape = input_shapes
        return (feat_shape[0], feat_shape[1], k, feat_shape[2])  # (N, P, K, C)

    return Lambda(gather_fn, output_shape=output_shape_fn)([topk_indices, features])

def edge_conv(points, features, num_points, K, channels, with_bn=True, activation='relu', pooling='average', name='edgeconv'):
    with tf.name_scope('edgeconv'):
        D = batch_distance_matrix_general(points, points)
        indices = Lambda(lambda x: tf.nn.top_k(-x, k=K + 1)[1])(D)  # Get indices only
        indices = Lambda(lambda x: x[:, :, 1:])(indices)

        fts = features
        knn_fts = knn(num_points, K, indices, fts)
        knn_fts_center = Lambda(lambda x: tf.tile(tf.expand_dims(x, axis=2), (1, 1, K, 1)))(fts)
        knn_fts = Lambda(lambda x: tf.concat([x[0], x[1] - x[0]], axis=-1))([knn_fts_center, knn_fts])

        x = knn_fts
        for idx, channel in enumerate(channels):
            x = tf.keras.layers.Conv2D(channel, kernel_size=(1, 1), strides=1, data_format='channels_last',use_bias=not with_bn, kernel_initializer='glorot_normal', name=f'{name}_conv{idx}')(x)
            if with_bn:
                x = BatchNormalization(name=f'{name}_bn{idx}')(x)
            if activation:
                x = tf.keras.layers.Activation(activation, name=f'{name}_act{idx}')(x)

        fts = Lambda(lambda x: tf.reduce_max(x, axis=2) if pooling == 'max' else tf.reduce_mean(x, axis=2))(x)

        shortcut = Lambda(lambda x: tf.expand_dims(x, axis=2))(features)
        sc = tf.keras.layers.Conv2D(channels[-1], kernel_size=(1, 1), strides=1, data_format='channels_last',use_bias=not with_bn, kernel_initializer='glorot_normal', name=f'{name}_sc_conv')(shortcut)
        if with_bn:
            sc = BatchNormalization(name=f'{name}_sc_bn')(sc)
        sc = Lambda(lambda x: tf.squeeze(x, axis=2))(sc)

        return tf.keras.layers.Activation(activation, name=f'{name}_sc_act')(sc + fts) if activation else sc + fts

class MaskProcessingLayer(Layer):
    def call(self, mask):
        mask = tf.cast(tf.not_equal(mask, 0), dtype='float32')
        coord_shift = 999. * tf.cast(tf.equal(mask, 0), dtype='float32')
        return mask, coord_shift

def _particle_net_base(points, features=None, mask=None, setting=None, name='particle_net'):
    if features is None:
        features = points

    if mask is not None:
        mask, coord_shift = MaskProcessingLayer()(mask)

    expanded = Lambda(lambda x: tf.expand_dims(x, axis=2), name = "expand_dim")(features)
    normalized = BatchNormalization(name=f'{name}_fts_bn')(expanded)
    fts = Lambda(lambda x: tf.squeeze(x, axis=2), name = "squeeze_norm")(normalized)

    for layer_idx, (K, channels) in enumerate(setting.conv_params):
        print(layer_idx)
        print(coord_shift.shape)
        print(points.shape)
        print(fts.shape)
        pts = Lambda(lambda x: tf.add(x[0], x[1]), name = f"tf.add{layer_idx}")([coord_shift, points if layer_idx == 0 else fts])
        fts = edge_conv(pts, fts, setting.num_points, K, channels, with_bn=True, activation='relu',pooling=setting.conv_pooling, name=f'{name}_EdgeConv{layer_idx}')

    if mask is not None:
        fts = Lambda(lambda x: x[0] * x[1], name="tf.multiply")([fts, mask])

    pool = Lambda(lambda x: tf.reduce_mean(x, axis=1))(fts)

    if setting.fc_params:
        x = pool
        for layer_idx, (units, drop_rate) in enumerate(setting.fc_params):
            x = tf.keras.layers.Dense(units, activation='relu')(x)
            if drop_rate:
                x = tf.keras.layers.Dropout(drop_rate)(x)
        out = tf.keras.layers.Dense(setting.num_class, activation='softmax')(x)
        return out
    else:
        return pool



class _DotDict:
    pass


def get_particle_net(num_classes, input_shapes):
    r"""ParticleNet model from `"ParticleNet: Jet Tagging via Particle Clouds"
    <https://arxiv.org/abs/1902.08570>`_ paper.
    Parameters
    ----------
    num_classes : int
        Number of output classes.
    input_shapes : dict
        The shapes of each input (`points`, `features`, `mask`).
    """
    setting = _DotDict()
    setting.num_class = num_classes
    # conv_params: list of tuple in the format (K, (C1, C2, C3))
    setting.conv_params = [
        (16, (64, 64, 64)),
        (16, (128, 128, 128)),
        (16, (256, 256, 256)),
        ]
    # conv_pooling: 'average' or 'max'
    setting.conv_pooling = 'average'
    # fc_params: list of tuples in the format (C, drop_rate)
    setting.fc_params = [(256, 0.1)]
    setting.num_points = input_shapes['points'][0]
    print(setting.num_points)
    points = tf.keras.Input(name='points', shape=input_shapes['points'])
    features = tf.keras.Input(name='features', shape=input_shapes['features']) if 'features' in input_shapes else None
    mask = tf.keras.Input(name='mask', shape=input_shapes['mask']) if 'mask' in input_shapes else None
    outputs = _particle_net_base(points, features, mask, setting, name='ParticleNet')

    return tf.keras.Model(inputs=[points, features, mask], outputs=outputs, name='ParticleNet')




# ------------------------------- DATASET -----------------------------------
def pad_array(a, maxlen, value=0., dtype='float32'):
    x = (np.ones((len(a), maxlen)) * value).astype(dtype)
    for idx, s in enumerate(a):
        if not len(s):
            continue
        trunc = s[:maxlen].astype(dtype)
        x[idx, :len(trunc)] = trunc
    return x


class Dataset(object):
    def __init__(self, filepath, feature_dict = {}, label='label', data_format='channel_first'):
        self.filepath = filepath
        self.feature_dict = feature_dict
        if len(feature_dict)==0:
            feature_dict['points'] = ['dEta', 'dPhi']
            feature_dict['features'] = ['log_pT', 'log_eT', 'ratio_log_pTj', 'ratio_log_eTj']
            feature_dict['mask'] = ['log_pT']
        self.label = label
        assert data_format in ('channel_first', 'channel_last')
        self.stack_axis = 1 if data_format=='channel_first' else -1
        self._values = {}
        self._label = None
        self._load()

    def _load(self):
        counts = None
        df = pd.read_csv(self.filepath)
        self._label = df[self.label].to_numpy()
        one_hot = np.zeros((self._label.shape[0], 2))
        one_hot[np.arange(self._label.shape[0]), self._label] = 1
        self._label = one_hot

        for k in self.feature_dict:
            cols = self.feature_dict[k]
            if not isinstance(cols, (list, tuple)):
                cols = [cols]
            arrs = []
            for col in cols:
                if counts is None:
                    counts = len(df[col])
                else:
                    assert counts == len(df[col])
                tcol = df[col].apply(ast.literal_eval)
                ele = np.array(tcol.to_list())
                # arrs.append(df[col].to_numpy())
                arrs.append(ele)
                # print(arrs[0][0])
            self._values[k] = np.stack(arrs, axis=self.stack_axis)

        print(f"Finished loading file {self.filepath}")

    def __len__(self):
        return len(self._label)

    def __getitem__(self, key):
        if key==self.label:
            return self._label
        else:
            return self._values[key]
    
    @property
    def X(self):
        return self._values
    
    @property
    def y(self):
        return self._label

    def shuffle(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        shuffle_indices = np.arange(self.__len__())
        np.random.shuffle(shuffle_indices)
        for k in self._values:
            self._values[k] = self._values[k][shuffle_indices]
        self._label = self._label[shuffle_indices]

def split_dataset(dataset, val_size=0.1, test_size=0.1, random_state=42):
    X = dataset.X
    y = dataset.y

    # First split into train+val and test
    indices = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(indices, test_size=test_size, stratify=y, random_state=random_state)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=val_size / (1 - test_size), stratify=y[idx_trainval], random_state=random_state)

    def create_subset(indices):
        sub_dataset = Dataset(dataset.filepath, dataset.feature_dict, dataset.label, 'channel_last')
        sub_dataset._label = y[indices]
        sub_dataset._values = {k: v[indices] for k, v in X.items()}
        return sub_dataset

    return create_subset(idx_train), create_subset(idx_val), create_subset(idx_test)

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 10:
        lr *= 0.1
    elif epoch > 20:
        lr *= 0.01
    return lr

if __name__ == "__main__":

    common_filepath = "../../output/pf_out"

    train_dataset= Dataset(f"{common_filepath}/train.csv", feature_dict={'points': ['dEta', 'dPhi'], 'features': ['log_pT', 'log_eT', 'ratio_log_pTj', 'ratio_log_eTj', 'dR'], 'mask': ['log_pT']}, label='label', data_format='channel_last')
    val_dataset= Dataset(f"{common_filepath}/val.csv", feature_dict={'points': ['dEta', 'dPhi'], 'features': ['log_pT', 'log_eT', 'ratio_log_pTj', 'ratio_log_eTj', 'dR'], 'mask': ['log_pT']}, label='label', data_format='channel_last')
    # test_dataset= Dataset(f"{common_filepath}/test.csv", feature_dict={'points': ['dEta', 'dPhi'], 'features': ['log_pT', 'log_eT', 'ratio_log_pTj', 'ratio_log_eTj', 'dR'], 'mask': ['log_pT']}, label='label', data_format='channel_last')
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    # print(f"Test size: {len(test_dataset)}")

    # print(train_dataset.y.shape)
    # for k in train_dataset.X:
    #     print(f"{k} :: {test_dataset[k].shape[1:]}")

    num_classes = val_dataset.y.shape[1]

    input_shapes = {k:val_dataset[k].shape[1:] for k in val_dataset.X}
    # print(val_dataset['points'][0])



    # # # ---model ---
    model = get_particle_net(num_classes, input_shapes)

    batch_size = 128
    epochs = 30

    model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule(0)),
              metrics=['accuracy'])
    model.summary()


    model_type = 'particle_net'
    save_dir = f"{common_filepath}/model_checkpoints"
    model_name = '%s_model.{epoch:03d}.keras' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,monitor='val_acc',verbose=1,save_best_only=True)

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    progress_bar = tf.keras.callbacks.ProgbarLogger()
    callbacks = [checkpoint, lr_scheduler, progress_bar]

    train_dataset.shuffle()
    model.fit(train_dataset.X, train_dataset.y,batch_size=batch_size,epochs=2,validation_data=(val_dataset.X, val_dataset.y),shuffle=True,callbacks=callbacks)