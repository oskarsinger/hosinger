from sklearn.decomposition import PCA

def get_whitened(data):

    pca = PCA(n_components=0.99999, whiten=True)

    return pca.fit_transform(data)

def get_pca_components(data):

    pca = PCA(n_components=0.9, whiten=True)

    return pca.fit(data).components_
