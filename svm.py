from sklearn import svm
import dataset
import pandas as pd


"""Loads dataset and splits into training/testing"""
def load_dataset(num_testing):
    X, y = dataset.load_dataset()

    train_X = X[:-num_testing]
    train_y = y[:-num_testing]    
    test_X = X[-num_testing:]
    test_y = y[-num_testing:]

    return train_X, train_y, test_X, test_y


"""Creates classifier and fits model"""
def create_model(X, y, gamma, kernel, C):
    clf = svm.SVC(gamma=gamma, kernel=kernel, C=C)
    clf.fit(X, y)
    return clf


"""Creates predictions and returns the number correct"""
def get_predictions(clf, test_X, test_y):
    predictions = clf.predict(test_X)
    return sum(int(a == y) for a, y in zip(predictions, test_y))
    

"""Runs SVM and prints out number of correct predictions"""
def run_svm(num_testing, gamma, kernel, C):
    train_X, train_y, test_X, test_y = load_dataset(num_testing)

    print 'Fitting model (training: %d, kernel: %s, gamma: %s, C: %.2f)' % (len(train_X) - num_testing, kernel, str(gamma), C)
    clf = create_model(train_X, train_y, gamma, kernel, C)

    print 'Testing predictions (testing: %d)' % ( num_testing)
    num_correct = get_predictions(clf, test_X, test_y)

    print 'Results: %d/%d correct (%d%%)' % (num_correct, num_testing, num_correct / (num_testing / 100))


if __name__ == '__main__':
    run_svm(num_testing=1428, gamma=0.01, kernel='rbf', C=10.0)
