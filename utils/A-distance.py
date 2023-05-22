# Compute A-distance using numpy and sklearn
# Reference: Analysis of representations in domain adaptation, NIPS-07.

import numpy as np
from sklearn import svm
import scipy.io as io


def proxy_a_distance(source_X, target_X, verbose=False):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]

    if verbose:
        print('PAD on', (nb_source, nb_target), 'examples')

    C_list = np.logspace(-5, 4, 10)

    half_source, half_target = int(nb_source/2), int(nb_target/2)
    train_X = np.vstack((source_X[0:half_source, :], target_X[0:half_target, :]))
    train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))

    test_X = np.vstack((source_X[half_source:, :], target_X[half_target:, :]))
    test_Y = np.hstack((np.zeros(nb_source - half_source, dtype=int), np.ones(nb_target - half_target, dtype=int)))

    best_risk = 1.0
    for C in C_list:
        clf = svm.SVC(C=C, kernel='linear', verbose=False)
        clf.fit(train_X, train_Y)

        train_risk = np.mean(clf.predict(train_X) != train_Y)
        test_risk = np.mean(clf.predict(test_X) != test_Y)

        if verbose:
            print('[ PAD C = %f ] train risk: %f  test risk: %f' % (C, train_risk, test_risk))

        if test_risk > .5:
            test_risk = 1. - test_risk

        best_risk = min(best_risk, test_risk)

    return 2 * (1. - 2 * best_risk)


def sample(data):
    count = data.shape[0]
    indexall = np.arange(count)
    np.random.shuffle(indexall)
    ted = int(count * 0.2)
    indexte = indexall[:ted]
    sample_data = data[indexte, :]
    return sample_data


if __name__ == '__main__':
    file_path = 'XXXX/'
    print(file_path)
    a_list = []
    for i in range(4):
        ### load source extracted features ###
        source_path = file_path + 'SCSN' + '_[' + str(i) + ']_feature_src.mat'
        matr = io.loadmat(source_path)
        source_data = matr["data"]
        source_data = sample(source_data)
        print(source_data.shape)
        ### load target extracted features ###
        target_path = file_path + 'SCSN' + '_[' + str(i) + ']_feature_tgt.mat'
        matr = io.loadmat(target_path)
        target_data = matr["data"]
        target_data = sample(target_data)
        # print(source_data.shape, target_data.shape)
        A = proxy_a_distance(source_data, target_data)
        a_list.append(A)
    for i in a_list:
        print(i)

