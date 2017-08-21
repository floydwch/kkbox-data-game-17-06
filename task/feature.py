# -*- coding: utf-8 -*-
from collections import Iterable
from functools import lru_cache
from datetime import datetime

from pandas import Series, DataFrame
from sklearn.feature_extraction import DictVectorizer

from .util import get_data


@lru_cache()
def get_title_encoder():
    (train_df,) = get_data('segment', ['train_df'])
    encoder = DictVectorizer(sparse=False)
    encoder.fit([dict(train_df['title_id'].value_counts())])
    return encoder


def get_title_counts(train_df):
    encoder = get_title_encoder()

    def titles(sample_df, ix):
        values = sample_df.loc[ix, 'title_id']
        if isinstance(values, Iterable):
            return values
        return Series([values])

    def title_counts(ix, sample_df, **kargs):
        return encoder.transform(
            [titles(sample_df, ix).value_counts()]
        )[0]

    return title_counts


def get_prefers(train_df, title_df):
    encoder = get_title_encoder()

    @lru_cache(None)
    def samples_quantile(title_id, quantile):
        return title_df.loc[title_id, 'watch_time'].quantile(quantile)

    def est_prefer(watch_time, title_id):
        if watch_time >= samples_quantile(title_id, .05):
            return 1
        elif samples_quantile(title_id, .01) < watch_time < \
                samples_quantile(title_id, .05):
            return -1
        else:
            return 0

    def prefers(ix, sample_df, title_df):
        user_df = sample_df.loc[ix]
        if isinstance(user_df, DataFrame):
            watch_times = user_df.groupby('title_id')['watch_time'].sum()
            preferences = {
                title_id: watch_time
                for title_id, watch_time in watch_times.items()
            }
            return encoder.transform([preferences])[0]
        else:
            return encoder.transform([{
                user_df['title_id']: user_df['watch_time']
            }])[0]

    return prefers


def get_event_time_medians(train_df):
    encoder = get_title_encoder()

    def event_time_medians(ix, sample_df, **kargs):
        selected_df = sample_df.loc[ix]
        if isinstance(selected_df, DataFrame):
            return encoder.transform(
                [selected_df.groupby('title_id')['time'].median()]
            )[0]
        else:
            return encoder.transform(
                [{selected_df['title_id']: selected_df['time']}]
            )[0]

    return event_time_medians


def get_event_time_max(train_df):
    encoder = get_title_encoder()

    def event_time_max(ix, sample_df, **kargs):
        selected_df = sample_df.loc[ix]
        if isinstance(selected_df, DataFrame):
            return encoder.transform(
                [selected_df.groupby('title_id')['time'].max()]
            )[0]
        else:
            return encoder.transform(
                [{selected_df['title_id']: selected_df['time']}]
            )[0]

    return event_time_max


def get_event_time_orders(train_df):
    encoder = get_title_encoder()

    def event_orders(ix, sample_df, **kargs):
        selected_df = sample_df.loc[ix]
        if isinstance(selected_df, DataFrame):
            orders = {
                key: i + 1
                for i, key in enumerate(
                    selected_df.groupby('title_id')[
                        'time'
                    ].median().sort_values().keys()
                )
            }
            return encoder.transform([orders])[0]
        else:
            return encoder.transform(
                [{selected_df['title_id']: 1}]
            )[0]

    return event_orders


def last_movie_watch_time(ix, sample_df, **kargs):
    encoder = get_title_encoder()
    selected_df = sample_df.loc[ix]

    if isinstance(selected_df, DataFrame):
        last = selected_df.sort_values('time').iloc[-1]
        feature = {
            last['title_id']: selected_df[
                selected_df['title_id'] == last['title_id']
            ]['watch_time'].sum()
        }
    else:
        feature = {selected_df['title_id']: selected_df['watch_time']}

    return encoder.transform([feature])[0]


def last_movie_time(ix, sample_df, **kargs):
    encoder = get_title_encoder()
    selected_df = sample_df.loc[ix]

    if isinstance(selected_df, DataFrame):
        last = selected_df.sort_values('time').iloc[-1]
        feature = {last['title_id']: last['time']}
    else:
        feature = {selected_df['title_id']: selected_df['time']}

    return encoder.transform([feature])[0]


def last_movie(ix, sample_df, **kargs):
    selected_df = sample_df.loc[ix]
    if isinstance(selected_df, DataFrame):
        last = selected_df.sort_values('time').iloc[-1]
        return last['title_id']
    else:
        return selected_df['title_id']


def epoch_to_hour(epoch):
    return datetime.fromtimestamp(epoch).hour


def event_time_hour(ix, sample_df, **kargs):
    selected_df = sample_df.loc[ix]
    if isinstance(selected_df, DataFrame):
        return selected_df['time'].apply(epoch_to_hour).median()
    else:
        return epoch_to_hour(selected_df['time'])


def epoch_to_weekday(epoch):
    return datetime.fromtimestamp(epoch).weekday()


def event_time_weekday(ix, sample_df, **kargs):
    selected_df = sample_df.loc[ix]
    if isinstance(selected_df, DataFrame):
        return selected_df['time'].apply(epoch_to_weekday).median()
    else:
        return epoch_to_weekday(selected_df['time'])


def total_watch_time(ix, sample_df, **kargs):
    selected_df = sample_df.loc[ix]
    if isinstance(selected_df, DataFrame):
        return selected_df['watch_time'].sum()
    else:
        return selected_df['watch_time']
