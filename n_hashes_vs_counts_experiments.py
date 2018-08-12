import numpy as np
from matplotlib import pyplot as plt

from lsh_functions import *
from synthetic_data_gen import *

# RANDOM SEED is set to 42 --- *** --- (hardcoded)
# This is now deterministic for this seed
# pass rng to every function that uses randomization as an argument
seed = 42
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
n_hashes_list = [1, 2, 3, 4, 5, 6]

print 'sigma: ', sigma
print 'c: ', c
print 't: ', t 
print 'w: ', w
print 'n_hashes: ', n_hashes_list
print

# how many iterations to run experiment
n_iter = 2

buckets_per_cluster = []
clusters_per_bucket = []

pts_list = []
clust_ids_list = [] 
for i in range(n_iter):
    pts, clust_ids, centers = generate_clusters(n, d, k, sigma, c, rng=rng)
    pts_list += [pts]
    clust_ids_list += [clust_ids]
print 'clusters generated'

for n_hashes in n_hashes_list:
    print 'n_hashes: ', n_hashes, ' start.'       
    bpc = 0
    cpb = 0
    for i in range(n_iter):
        pts = pts_list[i]
        clust_ids = clust_ids_list[i]
        
        full_hashes = run_plsh_l1(pts, t, w, n_hashes, mul=4, verbose=True, rng=rng)
        
        clusters, hash_counts = n_hashes_per_cluster(clust_ids, full_hashes)
        bpc += np.mean(hash_counts)
        
        buckets, clusters = clusters_in_buckets(clust_ids, full_hashes)
        bucket_counts = []
        for j in range(len(buckets)):
            bucket_counts += [np.unique(clusters[i]).shape[0]]
        cpb += np.mean(bucket_counts)
        print 'iteration :', i, ' done.'

    buckets_per_cluster += [bpc/float(n_iter)]
    clusters_per_bucket += [cpb/float(n_iter)]
    print 'n_hashes: ', n_hashes, ' done.'
    print

print 'buckets_per_cluster: ', buckets_per_cluster 
print 'clusters_per_bucket: ', clusters_per_bucket

title = 'n_buckets_per_cluster vs n_hashes'
plt.plot(n_hashes_list, buckets_per_cluster)
plt.grid(True)
plt.xlabel('n_hashes')
plt.ylabel('n_buckets')
plt.title(title)
plt.show()

title = 'n_clusters_per_bucket vs n_hashes'
plt.plot(n_hashes_list, clusters_per_bucket)
plt.grid(True)
plt.xlabel('n_hashes')
plt.ylabel('n_buckets')
plt.title(title)
plt.show()