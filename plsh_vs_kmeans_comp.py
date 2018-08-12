import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans

from lsh_functions import *
from synthetic_data_gen import *

# RANDOM SEED is set to 57 --- *** --- (hardcoded)
# This is now deterministic for this seed
# pass rng to every function that uses randomization as an argument
seed = 57
rng = np.random.RandomState(seed)

n = 100000
k = 100
d = 50

print 'n: ', n
print 'k: ', k
print 'd: ', d
print

sigma = 1
t = int(np.ceil(np.log(n)/float(np.power(np.log(np.log(n)), 2))))
w = 15
c = 20
n_hashes = 3

print 'sigma: ', sigma
print 'c: ', c
print 't: ', t 
print 'w: ', w
print 'n_hashes: ', n_hashes
print

pts, clust_ids, centers = generate_clusters(n, d, k, sigma, c, rng=rng)
print 'diameter of centers: ', np.max(pairwise_distances(centers))
print

# do plsh
mul = 4
print 'n_girds: ', int(np.power(2, mul*t*np.log(t)))
full_hashes = run_plsh_l1(pts, t, w, n_hashes, mul=mul, verbose=True, rng=rng)
print 

distinct_buckets = np.unique(full_hashes)
bucket_contents = []
for val in distinct_buckets:
    bucket_contents += [np.where(full_hashes==val)[0]]

seed = 0
rng = np.random.RandomState(seed)
# output a point from each bucket
S = []
for i in range(len(bucket_contents)):
    S += [rng.choice(bucket_contents[i], 1)[0]]

print 'k means objective approximation: ', find_kmeans_obj(pts, pts[S])

# now run actual k means
kmeans = KMeans(n_clusters=k, init='k-means++', n_init=5, max_iter=30, tol=100)
kmeans.fit(pts)
print 'k means objective: ',  kmeans.inertia_

print 'cost with some random k centers: ', find_kmeans_obj(pts, pts[rng.choice(np.arange(n), k, replace=False)])

print '--------------------------------------------------------------------------'