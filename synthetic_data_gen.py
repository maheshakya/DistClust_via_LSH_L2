import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.pairwise import pairwise_distances


rng = np.random.RandomState(42)

compare = lambda x, y: collections.Counter(x) == collections.Counter(y)

def dist_from_point_to_set(point, points_set, sep):
    if np.min(pairwise_distances(point.reshape(1, -1), points_set) >= sep):
        return True
    else:
        return False

def generate_centers(n_centers=5, dim=2, sep=10, rng=np.random.RandomState(42)):
    n_samples = n_centers*n_centers
    some_points = make_blobs(n_samples=n_samples, n_features=dim,
                             centers=1, cluster_std=sep/np.sqrt(dim), random_state=rng)
    centers = some_points[0][rng.randint(0, n_samples)].reshape(1, -1)
    while (len(centers) < n_centers):
        pt = some_points[0][rng.randint(0, n_samples)]
        if(dist_from_point_to_set(pt, centers, sep)):
            centers = np.vstack((centers, pt))
    return centers

def create_ratios(n, k, rng=np.random.RandomState(42)):
    a = rng.rand(k)
    m = n/np.sum(a)
    ratios = a*m
    ratios = np.array(ratios, dtype=int)
    ratios[np.argmin(ratios)] += 1
    diff = n - np.sum(ratios)
    ratios[np.argmax(ratios)] += + diff
    return ratios

def generate_clusters(n, d, k, sigma, c, rng=np.random.RandomState(42)):
    ratios = create_ratios(n, k, rng)
    centers = generate_centers(n_centers=k, dim=d, sep=c*sigma, rng=rng)    
    # set cluster std to for cluster radius = \sigma
    cluster_std = sigma/(2*np.sqrt(d))
    pts, clust_ids = make_blobs(ratios[0], d, centers=[centers[0]], cluster_std=cluster_std/float(rng.rand()), random_state=rng)
    for i in range(1, k):
        pt, _ = make_blobs(ratios[i], d, centers=[centers[i]], cluster_std=cluster_std, random_state=rng)
        pts = np.vstack((pts, pt))
        clust_ids = np.hstack((clust_ids, np.ones(ratios[i])*i))
    rnd_order = rng.choice(np.arange(n), n, replace=False)
    pts = pts[rnd_order]
    clust_ids = np.array(clust_ids[rnd_order], dtype=int)
    return pts, clust_ids, centers