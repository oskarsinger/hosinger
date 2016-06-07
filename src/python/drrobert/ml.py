from sklearn.decomposition import PCA

def get_pca(data, n_components=None, whiten=False):

    pca = PCA(n_components=n_components, whiten=whiten)

    return pca.fit_transform(data)

def get_pca_components(data):

    pca = PCA(n_components=0.9, whiten=True)

    return pca.fit(data).components_
