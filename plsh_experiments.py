import numpy as np

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

# evaluation
# ---------------------

print "-----------evaluations -----------"
print
clusters, hash_counts = n_hashes_per_cluster(clust_ids, full_hashes)
print 'avg # of buckets per cluster: ', np.mean(hash_counts)

buckets, clusters = clusters_in_buckets(clust_ids, full_hashes)
print 'total # of buckets: ', buckets.shape[0]

bucket_stats = []
bucket_counts = []
for i in range(len(buckets)):
    a, b = np.unique(clusters[i], return_counts=True)
    order = np.argsort(b)[::-1]
    bucket_stats += [[a[order], b[order]]]
    bucket_counts += [a.shape[0]]

print 'avg # of clusters in a bucket: ', np.mean(bucket_counts)
print 

# contents in the buckets
print 'contents in buckets - ordered'
for i in range(len(bucket_stats)):
    print 'bucket: ', buckets[i], '  clusters: ', bucket_stats[i][0], '  counts: ', bucket_stats[i][1]
print

# avg # of pts in a bucket
distinct_buckets = np.unique(full_hashes)
bucket_counts = []
for val in distinct_buckets:
    bucket_counts += [np.where(full_hashes==val)[0].shape[0]]
print 'avg pts in a bucket: ', np.mean(bucket_counts)
print 'max pts in a bucket: ', np.max(bucket_counts)
print 'min pts in a bucket: ', np.min(bucket_counts)

print '---------------------------------'



