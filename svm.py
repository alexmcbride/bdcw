from sklearn import svm
import dataset
import pandas as pd


"""Runs SVM and prints out number of correct predictions"""
def run_svm(num_testing, gamma, kernel, C):
    # Load dataset
    X, y = dataset.load_dataset()
    train_X = X[:-num_testing]
    train_y = y[:-num_testing]    
    test_X = X[-num_testing:]
    test_y = y[-num_testing:]

    # print 'Fitting model (training: %d, kernel: %s, gamma: %s, C: %.2f)' % (len(train_X) - num_testing, kernel, str(gamma), C)
    clf = svm.SVC(gamma=gamma, kernel=kernel, C=C)
    clf.fit(train_X, train_y)

    # print 'Testing predictions (testing: %d)' % ( num_testing)
    predictions = clf.predict(test_X)
    predictions = zip(predictions, test_y)
    num_correct = sum(int(a == y) for a, y in predictions)

    print 'Results: %d/%d correct (%.0f%%)' % (num_correct, num_testing, num_correct / (num_testing / 100.0))


if __name__ == '__main__':
    run_svm(num_testing=1428, gamma=0.01, kernel='rbf', C=10.0)
