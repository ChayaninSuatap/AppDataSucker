from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
from sklearn.decomposition import PCA

def kmeans_clustering(data, k):
    centroids,_ = kmeans(data, k)
    idx, _ = vq(data, centroids)

    return idx

def make_pca(data, dim=2):
    pca = PCA(n_components=dim)
    result = pca.fit_transform(data)
    return result

if __name__ == '__main__':
    pass

