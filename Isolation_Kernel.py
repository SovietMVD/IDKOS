import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix, hstack

def IKFeature(data, Sdata=None, psi=0, t=0, Sp=True):
    if Sdata is None:
        Sdata = data

    sizeS = Sdata.shape[0]
    sizeN = data.shape[0]
    features = []
    np.random.seed(0)
    for i in range(t):
        subIndex = np.random.choice(sizeS, psi, replace=False)
        tdata = Sdata[subIndex, :]
        tree = KDTree(tdata)
        _, nn_idx = tree.query(data, k=1)
        row = np.arange(sizeN)
        col = nn_idx.flatten()
        one_feature = csr_matrix((np.ones(sizeN), (row, col)), shape=(sizeN, psi))
        features.append(one_feature)

    feature_matrix = hstack(features)

    if Sp:
        return feature_matrix
    else:
        return feature_matrix.toarray()

def IKSimilarity(data, Sdata=None, psi=64, t=200):
    feature_matrix = IKFeature(data, Sdata, psi, t,True)
    similarity_matrix = (feature_matrix @ feature_matrix.T)/ t
    row_averages = similarity_matrix.mean(axis=1).A1
    total_average = row_averages.mean()
    low_avg_rows = [(i, avg) for i, avg in enumerate(row_averages) if avg <= total_average-0.03]
    indexs = [pair[0] for pair in low_avg_rows[:]]
    return row_averages,indexs

