# -*- coding: utf-8 -*-
import pickle
from collections import Iterable


def dump_to_pickle(data, name):
    with open('{}.pickle'.format(name), 'wb') as file:
        pickle.dump(data, file)


def load_from_pickle(name):
    with open('{}.pickle'.format(name), 'rb') as file:
        return pickle.load(file)


def get_data(name, items):
    data = load_from_pickle(name)
    return (data[item] for item in items)


def to_list(values):
    if isinstance(values, Iterable):
        return values
    return [values]
