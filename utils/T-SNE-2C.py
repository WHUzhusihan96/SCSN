import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import scipy.io as io

class FeatureVisualize(object):
    '''
    Visualize features by TSNE
    '''

    def __init__(self, data, label, count=0):
        '''
        features: (m,n)
        labels: (m,)
        '''
        # data and label contain both source and target domain
        # and we use count to seperate it
        self.data = data
        self.label = label
        self.source_count = count

    def plot_tsne_doubleColor(self, filename, save_eps=False):
        ''' Plot TSNE figure. Set save_eps=True if you want to save a .eps file.
        '''
        # compute all
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        features = tsne.fit_transform(self.data)
        x_min, x_max = np.min(features, 0), np.max(features, 0)
        data = (features - x_min) / (x_max - x_min)
        del features
        source_data = data[:source_count, :]
        target_data = data[source_count:, :]
        plt.scatter(source_data[:, 0], source_data[:, 1], c='#4169E1', marker='o')
        plt.scatter(target_data[:, 0], target_data[:, 1], c='#DC143C', marker='o')

        plt.xticks([])
        plt.yticks([])
        # plt.title('T-SNE')
        if save_eps:
            plt.savefig(filename, dpi=600, format='eps')
        plt.show()


def sample(data, label):
    count = data.shape[0]
    indexall = np.arange(count)
    np.random.shuffle(indexall)
    ted = int(count * 0.4)
    indexte = indexall[:ted]
    sample_data = data[indexte, :]
    sample_label = label[indexte]
    return sample_data, sample_label


if __name__ == '__main__':
    ### load source extracted features ###
    path = 'XXXX'
    alg = 'SCSN'
    list_number = [[0], [1], [2], [3]]
    src_path = []
    tgt_path = []
    for i in range(4):
        temp = path + alg + '_' + str(list_number[i]) + '_feature_src.mat'
        src_path.append(temp)
        temp = path + alg + '_' + str(list_number[i]) + '_feature_tgt.mat'
        tgt_path.append(temp)


    for i in range(4):
        matr = io.loadmat(src_path[i])
        source_data = matr["data"]
        source_label = matr["label"]
        source_label = source_label.flatten()
        # source_data, source_label = sample(source_data, source_label)

        source_count = source_data.shape[0]

        matr1 = io.loadmat(tgt_path[i])
        target_data = matr1["data"]
        target_label = matr1["label"]
        target_label = target_label.flatten()
        # target_data, target_label = sample(target_data, target_label)

        data = np.vstack((source_data, target_data))
        label = np.hstack((source_label, target_label))

        vis = FeatureVisualize(data, label, source_count)
        ### plot and save the result ###
        save_path = path + alg + '_' + str(list_number[i]) + '-vis-2c.eps'
        vis.plot_tsne_doubleColor(save_path, save_eps=True)
