from sklearn.model_selection import cross_val_score
from sklearn import svm
import pandas as pd
import numpy as np
import pickle


"""Loads dataset, pre-processes it, then pickles it. This should be run 
first before any other scripts."""
def clean_dataset():
    # Load CSV file.
    df = pd.read_csv('mushroom.csv')

    # Remove records with missing values
    df.dropna(inplace=True)

    # Randomly shuffle the dataset to remove any ordering bias
    df = df.sample(frac=1).reset_index(drop=True)

    # Remove columns not wanted in the data
    df = df.drop(columns=['veil-type'])

    # Combine categories that comprise < 5% of total into single category labeled 'z'
    columns = list(df.columns)
    total = len(df.index)
    for column in columns:
        value_counts = df[column].value_counts()
        combine = []
        for label, count in value_counts.iteritems():
            if count / (total / 100 ) < 5:
                combine.append(label)
        for label in combine:
            df[column].replace(label, 'z', inplace=True)

    # Convert from categorical to numerical
    df = pd.get_dummies(df)

    # Save to disk
    print 'Saving cleaned dataset to disk'
    with open('data', 'wb') as file:
        pickle.dump(df, file)
    print 'Done!'


"""Loads pickled dataset and returns X (training) and y (testing) as a tuple"""
def load_dataset():
    # Open data file
    with open('data', 'rb') as file:
        # Load data
        df = pickle.load(file)

        # Get testing data
        y = np.ravel(df['Class_p'])

        # Get rid of class from training data
        X = df.drop(columns=['Class_p', 'Class_e'])

        return (X, y)


if __name__ == '__main__':
    pass
    clean_dataset()
