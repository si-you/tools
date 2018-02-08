from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv

from absl import app
from absl import flags

import pandas as pd


FLAGS = flags.FLAGS


flags.DEFINE_string('dataset', None, 'A path to the dataset.')
flags.DEFINE_float('test_fraction', 0.2, 'A split fraction between [0.0, 1.0]')


def main(unused_args):
    df = pd.read_csv(FLAGS.dataset, sep=',', dtype=object,
                     quoting=csv.QUOTE_NONE)

    # Step 1. Shuffle the dataset.
    df = df.sample(frac=1.0).reset_index(drop=True)
    split_point = int(len(df) * FLAGS.test_fraction)

    # Step 2. Split the dataset.
    df_test = df.iloc[:split_point, :]
    df_train = df.iloc[split_point:, :]

    # Step 3. Materialize the datasets.
    df_train.to_csv('train.csv', sep=',', quoting=csv.QUOTE_NONE,
                    index=False)
    df_test.to_csv('test.csv', sep=',', quoting=csv.QUOTE_NONE,
                    index=False)


if __name__ == '__main__':
    app.run(main)
