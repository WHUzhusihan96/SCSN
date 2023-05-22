import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import scipy.io as io
import matplotlib.colors as col


def colormap():
    return col.LinearSegmentedColormap.from_list('cmap', ['#FF0000', '#FFFF00', '#00FF00', '#FFA500',
                                                          '#00FFFF', '#4169E1', '#EE82EE'], 256)
    # red, yellow, lime, cyan, royalblue, voilet
    # ['#A52A2A', '#FFD700', '#008000', '#FFA500', '#4169E1', '#800080']
    # ['#FF0000', '#FFFF00', '#00FF00', '#00FFFF', '#4169E1', '#EE82EE']
def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


class FeatureVisualize(object):
    '''
    Visualize features by TSNE
    '''

    def __init__(self, features, labels, count):
        '''
        features: (m,n)
        labels: (m,)
        '''
        self.features = features
        self.labels = labels
        self.source_count = count

    def plot_tsne(self, path, save_eps=False):
        ''' Plot TSNE figure. Set save_eps=True if you want to save a .eps file.
        '''
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        features = tsne.fit_transform(self.features)
        x_min, x_max = np.min(features, 0), np.max(features, 0)
        data = (features - x_min) / (x_max - x_min)
        del features
        # define the color and type
        m = {0: 'o', 1: '^', 2: 's', 3: '*', 4: 'D', 5: 'P', 6: 'X'}
        # plot
        fig, ax = plt.subplots()

        source_data = data[:self.source_count, :]
        source_label = label[:self.source_count]
        cm1 = list(map(lambda x: m[x], source_label))  # 将相应的标签改为对应的marker
        # scatter = mscatter(source_data[:, 0], source_data[:, 1], c=source_label, m=cm1, ax=ax, cmap=colormap())

        target_data = data[self.source_count:, :]
        target_label = label[self.source_count:]
        cm2 = list(map(lambda x: m[x], target_label))  # 将相应的标签改为对应的marker
        scatter = mscatter(target_data[:, 0], target_data[:, 1], c=target_label, m=cm2, ax=ax, cmap=colormap())

        plt.xticks([])
        plt.yticks([])
        # plt.title('T-SNE')
        if save_eps:
            plt.savefig(path, dpi=600, format='eps')
        plt.show()


def sample(data, label):
    count = data.shape[0]
    indexall = np.arange(count)
    np.random.shuffle(indexall)
    ted = int(count * 0.5)
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
        save_path = path + alg + '_' + str(list_number[i]) + '-vis-mc-tgtall.eps'
        vis.plot_tsne(save_path, save_eps=True)
