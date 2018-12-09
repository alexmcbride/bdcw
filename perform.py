import timeit

def go(number):
    result = timeit.timeit("svm.run_svm(num_testing=1428, gamma=0.01, kernel='rbf', C=1.0)", setup='import svm', number=number)
    print 'C: 1.0 - %.2f seconds' % (result / number)    
    result = timeit.timeit("svm.run_svm(num_testing=1428, gamma=0.01, kernel='rbf', C=10.0)", setup='import svm', number=number)
    print 'C: 10.0 - %.2f seconds' % (result / number)
    result = timeit.timeit("svm.run_svm(num_testing=1428, gamma=0.01, kernel='rbf', C=100.0)", setup='import svm', number=number)
    print 'C: 100.0 - %.2f seconds' % (result / number)
    
go(10)