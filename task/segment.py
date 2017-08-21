# -*- coding: utf-8 -*-
import os
import datetime
import calendar

from pandas import read_csv, DataFrame
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from .util import dump_to_pickle


if __name__ == '__main__':
    train_df, test_df = map(
        lambda category:
            read_csv(
                'data/events_{}.csv'.format(category),
                index_col='user_id'
            ).sort_index(),
        ('train', 'test')
    )
    train_labels = read_csv(
        'data/labels_train.csv',
        index_col='user_id'
    )
    all_train_indices, test_indices = map(
        lambda df: list(set(df.index.values)),
        (train_df, test_df)
    )
    train_indices, val_indices = train_test_split(
        all_train_indices,
        test_size=.1,
        random_state=0
    )
    test_indices = sorted(test_indices)
    train_targets = train_labels.loc[train_indices, 'title_id'].values

    output = dict(
        train_df=train_df,
        train_labels=train_labels,
        train_indices=train_indices,
        train_targets=train_targets,
        val_indices=val_indices,
        val_targets=train_labels.loc[val_indices, 'title_id'].values,
        test_df=test_df,
        test_indices=test_indices
    )

    dump_to_pickle(output, 'segment')

    if os.path.isfile('extract.pickle'):
        os.remove('extract.pickle')

    if os.path.isfile('feature.pickle'):
        os.remove('feature.pickle')
