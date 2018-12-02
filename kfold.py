from sklearn.model_selection import cross_val_score
from sklearn import svm
import dataset


"""Runs k-fold validation on the dataset using the specified params"""
def run_kfold(gamma='auto', kernel='linear', C=1.0, k=3):
    print 'Running k-fold cross-validation for mushroom dataset'
    print 'Kernel: %s, C: %0.1f, gamma: %s, for %d folds' % (kernel, C, str(gamma), k)

    # Get dataset
    X, y = dataset.load_dataset()
    clf = svm.SVC(gamma=gamma, kernel=kernel, C=C)

    # Use k-fold validation to get accuracy of the model.
    scores = cross_val_score(clf, X, y, cv=k)
    print 'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)


if __name__ == '__main__':
    run_kfold(gamma=0.01, kernel='rbf', C=10.0, k=10)
                                                        