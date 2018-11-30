from sklearn import svm
import dataset


"""Runs SVM and prints out number of correct predictions"""
def run_svm(num_testing=1428):
    # Get dataset and split into training/testing
    data, target = dataset.load_dataset()

    # Split into training and testing.
    testing_data = data[-num_testing:]
    testing_target = target[-num_testing:]
    data = data[:-num_testing]
    target = target[:-num_testing]

    # Create SVC and fit data.
    clf = svm.SVC(gamma='auto', kernel='rbf', C=10.0)
    clf.fit(data, target)

    # Get predictions
    predictions = clf.predict(testing_data)
    num_correct = sum(int(a == y) for a, y in zip(predictions, testing_target))

    print '%d out of %d correct (%d percent)' % (num_correct, num_testing, num_correct / (num_testing / 100))


if __name__ == '__main__':
    run_svm()