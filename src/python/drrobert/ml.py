from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def get_kmeans(data, k=2):

    model = KMeans(
        n_clusters=k,
        random_state=0)

    model.fit(data)

    return model

def get_pca(data, n_components=None, whiten=False):

    pca = PCA(n_components=n_components, whiten=whiten)

    return pca.fit_transform(data)

def get_pca_components(data):

    pca = PCA(n_components=0.9, whiten=True)

    return pca.fit(data).components_
