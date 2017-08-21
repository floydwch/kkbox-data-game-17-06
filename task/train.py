# -*- coding: utf-8 -*-
import os
from itertools import chain
from datetime import datetime

import numpy as np
from pandas import Series, DataFrame
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import (
    cross_val_score, KFold, train_test_split
)
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
        LabelEncoder, RobustScaler, normalize
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, BatchNormalization, Dropout, AlphaDropout, Merge, Concatenate
from keras.regularizers import l1, l2, l1_l2
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.activations import selu

from .util import get_data, dump_to_pickle, to_list


def to_8d_str(value):
    return '{:08d}'.format(value)


def get_features(selected_feature_names):
    if os.path.isfile('feature.pickle'):
        return tuple(
            get_data(
                'feature',
                ['val_features', 'test_features', 'train_features']
            )
        )

    val_feature_df, test_feature_df, train_feature_df = get_data(
        'extract',
        ['val_feature_df', 'test_feature_df', 'train_feature_df']
    )

    val_features, test_features, train_features = map(
        lambda df:
            np.array(df[selected_feature_names].apply(
                lambda row: Series(
                    list(chain.from_iterable(map(to_list, row)))
                ),
                1
            )).astype(float),
        (val_feature_df, test_feature_df, train_feature_df)
    )

    dump_to_pickle(dict(
        val_features=val_features,
        test_features=test_features,
        train_features=train_features
    ), 'feature')

    return val_features, test_features, train_features


if __name__ == '__main__':
    val_features, test_features, train_features = get_features(
        [
            'last_movie_time',
            'prefers',
            'event_time_max',
            'event_time_hour',
            'event_time_weekday',
            'total_watch_time'
        ]
    )
    (val_targets, train_targets) = get_data(
        'segment', ['val_targets', 'train_targets']
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(np.hstack([train_targets, val_targets]))

    n_classes = np.unique(np.hstack([train_targets, val_targets])).shape[0]

    model1 = Sequential()
    model1.add(Dense(
        n_classes,
        input_shape=(414,),
        kernel_initializer='lecun_normal'
    ))
    model1.add(BatchNormalization())
    model1.add(Activation('selu'))
    model1.add(Dense(
        n_classes,
        kernel_initializer='lecun_normal'
    ))
    model1.add(BatchNormalization())
    model1.add(Activation('selu'))

    model2 = Sequential()
    model2.add(Dropout(.2, input_shape=(2 * 414 + 3,)))
    model2.add(Dense(
        2 * 414 + 3,
        kernel_initializer='lecun_normal',
        kernel_regularizer=l2(),
        bias_regularizer=l2()
    ))
    model2.add(BatchNormalization())
    model2.add(Activation('selu'))
    model2.add(Dropout(.2))
    model2.add(Dense(
        n_classes,
        kernel_initializer='lecun_normal',
        kernel_regularizer=l2(),
        bias_regularizer=l2()
    ))
    model2.add(BatchNormalization())
    model2.add(Activation('selu'))
    model2.add(Dropout(.2))
    model2.add(Dense(
        n_classes,
        kernel_initializer='lecun_normal',
        kernel_regularizer=l2(),
        bias_regularizer=l2()
    ))
    model2.add(BatchNormalization())
    model2.add(Activation('selu'))

    model3 = Sequential()
    model3.add(Dropout(.4, input_shape=(2 * 414 + 3,)))
    model3.add(Dense(
        2 * 414 + 3,
        kernel_initializer='lecun_normal',
        kernel_regularizer=l2(),
        bias_regularizer=l2()
    ))
    model3.add(BatchNormalization())
    model3.add(Activation('selu'))
    model3.add(Dropout(.4))
    model3.add(Dense(
        n_classes,
        kernel_initializer='lecun_normal',
        kernel_regularizer=l2(),
        bias_regularizer=l2()
    ))
    model3.add(BatchNormalization())
    model3.add(Activation('selu'))
    model3.add(Dropout(.4))
    model3.add(Dense(
        n_classes,
        kernel_initializer='lecun_normal',
        kernel_regularizer=l2(),
        bias_regularizer=l2()
    ))
    model3.add(BatchNormalization())
    model3.add(Activation('selu'))

    model4 = Sequential()
    model4.add(Dropout(.5, input_shape=(2 * 414 + 3,)))
    model4.add(Dense(
        2 * 414 + 3,
        kernel_initializer='lecun_normal',
        kernel_regularizer=l2(),
        bias_regularizer=l2()
    ))
    model4.add(BatchNormalization())
    model4.add(Activation('selu'))
    model4.add(Dropout(.5))
    model4.add(Dense(
        n_classes,
        kernel_initializer='lecun_normal',
        kernel_regularizer=l2(),
        bias_regularizer=l2()
    ))
    model4.add(BatchNormalization())
    model4.add(Activation('selu'))
    model4.add(Dropout(.5))
    model4.add(Dense(
        n_classes,
        kernel_initializer='lecun_normal',
        kernel_regularizer=l2(),
        bias_regularizer=l2()
    ))
    model4.add(BatchNormalization())
    model4.add(Activation('selu'))

    model234 = Sequential()
    model234.add(Merge([model2, model3, model4], mode='sum'))

    model = Sequential()
    model.add(Merge([model1, model234], mode='concat'))
    model.add(Dense(
        n_classes,
        kernel_initializer='zeros',
        bias_initializer='zeros'
    ))
    model.add(BatchNormalization())
    model.add(Activation('linear'))
    model.add(Dense(
        n_classes
    ))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='nadam',
        metrics=['accuracy']
    )

    train_targets = np_utils.to_categorical(
        label_encoder.transform(train_targets),
        n_classes
    )
    val_targets = np_utils.to_categorical(
        label_encoder.transform(val_targets),
        n_classes
    )

    model.fit(
        [
            np.vstack([train_features, val_features])[:, :414],
            np.vstack([train_features, val_features])[:, 414:],
            np.vstack([train_features, val_features])[:, 414:],
            np.vstack([train_features, val_features])[:, 414:]
        ],
        np.vstack([train_targets, val_targets]),
        epochs=50,
        batch_size=64,
        shuffle=True,
    )

    (test_indices,) = get_data('segment', ['test_indices'])
    (test_features,) = get_data('feature', ['test_features'])
    test_preds = label_encoder.inverse_transform(
        model.predict_classes([
            test_features[:, :414],
            test_features[:, 414:],
            test_features[:, 414:],
            test_features[:, 414:]
        ])
    )
    test_indices = map(to_8d_str, test_indices)
    test_preds = map(to_8d_str, test_preds)
    pred_df = DataFrame(
        list(zip(test_indices, test_preds)),
        columns=['user_id', 'title_id']
    )

    submission_dir = 'submission'
    if not os.path.exists(submission_dir):
        os.mkdir(submission_dir)

    time = datetime.now().strftime('%d-%H-%M')
    pred_df.to_csv('submission/{}.csv'.format(time), index=False)
