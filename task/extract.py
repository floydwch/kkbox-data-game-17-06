# -*- coding: utf-8 -*-
import os
from itertools import combinations, chain, starmap
from collections import Counter

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

from .util import get_data, dump_to_pickle, to_list
from .feature import (
    get_prefers, get_title_counts, get_event_time_medians,
    get_event_time_orders, get_event_time_max, last_movie_watch_time,
    last_movie_time, last_movie, event_time_hour, event_time_weekday,
    total_watch_time
)


def get_co_occur_marix(sample_df, labels):
    user_items = pd.concat(
        [sample_df['title_id'], labels['title_id']]
    ).sort_index()
    counter = Counter(chain.from_iterable(
        combinations(set(to_list(user_items[ix])), 2)
        for ix in user_items.index.drop_duplicates()
    ))
    label_encoder = LabelEncoder()
    label_types = np.unique(user_items.values)
    label_encoder.fit(label_types)
    n_row = label_types.size
    matrix = np.zeros((n_row, n_row))
    for key, value in counter.items():
        matrix[
            label_encoder.transform([key[0]])[0],
            label_encoder.transform([key[1]])[0]
        ] = value
    return matrix


if __name__ == '__main__':
    (
        train_df, train_labels, train_indices,
        val_indices,
        test_df, test_indices
    ) = get_data('segment', [
        'train_df', 'train_labels', 'train_indices',
        'val_indices',
        'test_df', 'test_indices'
    ])

    sample_df = pd.concat([train_df, test_df]).sort_index()
    title_df = sample_df.set_index('title_id').sort_index()

    feature_fns = [
        last_movie_time,
        get_prefers(train_df, title_df),
        get_event_time_max(train_df),
        event_time_hour,
        event_time_weekday,
        total_watch_time,
    ]

    val_feature_df, test_feature_df, train_feature_df = starmap(
        lambda indices, sample_df: DataFrame.from_dict(
            dict(map(
                lambda fn: (
                    fn.__name__, list(map(
                        lambda ix: fn(
                            ix,
                            sample_df=sample_df,
                            title_df=title_df
                        ),
                        indices
                    ))
                ),
                feature_fns
            ))
        ),
        (
            (val_indices, sample_df),
            (test_indices, sample_df),
            (train_indices, sample_df)
        )
    )

    dump_to_pickle(dict(
        val_feature_df=val_feature_df,
        test_feature_df=test_feature_df,
        train_feature_df=train_feature_df
    ), 'extract')

    if os.path.isfile('feature.pickle'):
        os.remove('feature.pickle')
