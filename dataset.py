from sklearn.model_selection import cross_val_score
from sklearn import svm
import pandas as pd
import numpy as np


"""Loads dataset and returns data and target class as a tuple e.g. (data, target)."""
def load_dataset():
    # Load CSV file.
    df = pd.read_csv('mushroom.csv')

    # Remove records with missing values
    df.dropna(inplace=True)

    # Randomly shuffle the dataset to remove any ordering bias
    df = df.sample(frac=1).reset_index(drop=True)

    # Get target data
    target = np.ravel([int(c == 'p') for c in df['Class']])

    # Get rid of class from training data
    data = df.drop(columns=['Class'])

    # Convert from categorical to numerical
    data = pd.get_dummies(data)

    return (data, target)


"""Generates all the CSV files used to make excel charts"""
def generate_charts():
    df = pd.read_csv('mushroom.csv')
    columns = df.columns.values
    for column in columns:
        output = get_counts_csv(column)
        with open('charts/%s.csv' % (column), 'w') as f:
            f.write(output)


"""Helper that prints out a CSV of attribute values, that can then be used to generate charts in Excel"""
def get_counts_csv(attribute='cap-shape', drop_missing=False):
    data = pd.read_csv('mushroom.csv')
    if drop_missing:
        data.dropna(inplace=True)
    p = dict(data[data['Class'] == 'p'][attribute].value_counts().iteritems())
    e = dict(data[data['Class'] == 'e'][attribute].value_counts().iteritems())
    output = 'Name,Poisonous,Edible\n'
    used = []
    for label, count in p.iteritems():
        e_count = e.get(label)
        output += '%s,%d,%d\n' % (label, count, e_count if e_count else 0)
        used.append(label)
    for label, count in e.iteritems():
        if label not in used:
            p_count = p.get(label)
            output += '%s,%d,%d\n' % (label, p_count if p_count else 0, count)
    return output


def stuff():
    df = pd.read_csv('mushroom.csv')

    # Remove records with missing values
    df.dropna(inplace=True)

    df = df.drop(columns=['Class'])

    df = pd.get_dummies(df)

    print df.head()


if __name__ == '__main__':
    # generate_charts()
    stuff()

