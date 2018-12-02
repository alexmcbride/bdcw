from sklearn.model_selection import cross_val_score
from sklearn import svm
import pandas as pd
import numpy as np
import pickle


"""Loads dataset, pre-processes it, then pickles it"""
def clean_dataset():
    # Load CSV file.
    df = pd.read_csv('mushroom.csv')

    # Remove records with missing values
    df.dropna(inplace=True)

    # Randomly shuffle the dataset to remove any ordering bias
    df = df.sample(frac=1).reset_index(drop=True)

    # Remove columns not wanted in the data
    df = df.drop(columns=['veil-type'])

    # Convert from categorical to numerical
    df = pd.get_dummies(df)

    with open('data', 'wb') as file:
        pickle.dump(df, file)


"""Loads pickled dataset and returns X and y as a tuple"""
def load_dataset():
    with open('data', 'rb') as file:
        df = pickle.load(file)

        # Get testing data
        y = np.ravel(df['Class_p'])

        # Get rid of class from training data
        X = df.drop(columns=['Class_p', 'Class_e'])

        return (X, y)


if __name__ == '__main__':
    clean_dataset()
