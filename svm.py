from sklearn import svm
import dataset


"""Runs SVM and prints out number of correct predictions"""
def run_svm(gamma='auto', kernel='rbf', C=1.0, num_testing=1428):
    # Get dataset and split into training/testing
    data, target = dataset.load_dataset()

    # Split into training and testing.
    testing_data = data[-num_testing:]
    testing_target = target[-num_testing:]
    data = data[:-num_testing]
    target = target[:-num_testing]

    # Create SVC
    print 'Creating SVC model (gamma: %s, kernel: %s, C: %.1f)' % (gamma, kernel, C)    
    clf = svm.SVC(gamma=gamma, kernel=kernel, C=C)

    # Fit data
    print 'Fitting model on training data (%d records)' % (len(data) - num_testing)
    clf.fit(data, target)

    # Get predictions
    print 'Generating predictions on testing data (%d records)' % ( num_testing)
    predictions = clf.predict(testing_data)
    num_correct = sum(int(a == y) for a, y in zip(predictions, testing_target))
    print 'Results: %d/%d correct (%d percent)' % (num_correct, num_testing, num_correct / (num_testing / 100))


if __name__ == '__main__':
    run_svm(kernel='rbf', C=10.0, num_testing=1428)
