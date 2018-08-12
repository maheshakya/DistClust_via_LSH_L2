import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.pairwise import pairwise_distances
import hashlib

def get_hash_matrix(t=5, d=10, rng=np.random.RandomState(42)):
    return rng.randn(t, d)

def find_closest(val, w):
    a = val/float(4*w)
    a_ = np.floor(a)
    if a - a_ >= 0.5:
        return a_+ 1
    else:
        return a_ 

def generate_grids(t=5, multiplier=2, rng=np.random.RandomState(42)):
    """
    low and high: boundries
    # of grids =  2^(multiplier * (tlogt))
    """
    shifts = rng.rand(int(np.power(2, multiplier*t*np.log(t))), t)
    return shifts

def calc_ball_in_grid(pt, shift, w):
        t = pt.shape[0]
        pt_ = pt - (shift * 4 * w)
        closest_pt_scaled = []
        for j in range(t):
            closest_pt_scaled += [int(find_closest(pt_[j], w))]
        if np.linalg.norm(pt_ - np.array(closest_pt_scaled) * 4 * w) <= w:
            return True, closest_pt_scaled
        return False, None
    
def calc_hash(pt, shifts, w):
    """
    assuming axis cordinates are always symmetric around origin
    """
    t = pt.shape[0]
    n_shifts = shifts.shape[0]
    hash_val = ['a']
    for i in range(t):
        hash_val += ['a']
    for i in range(n_shifts):
        is_true, grid_center = calc_ball_in_grid(pt, shifts[i], w)
        if is_true:
            hash_val = [i] + grid_center
            return hash_val
    return hash_val

def make_string_hash(hashed_ball):
#    return ','.join(map(str, hashed_ball)).replace(',', '')
    return ','.join(map(str, hashed_ball))

    
def multi_hash(hashed_points, shifts, w):
    """returns the concatenated hash as a string"""
    full_hash = []
    for i in range(len(hashed_points)):
        full_hash += [make_string_hash(calc_hash(hashed_points[i], shifts, w))]
    return make_string_hash(full_hash)

def run_plsh_l1(pts, t, w, n_hashes, mul=3, verbose=False, verbose_denom=10, rng=np.random.RandomState(42)):
    n = pts.shape[0]
    d = pts.shape[1]
    
    #create grids
    shifts = generate_grids(t, multiplier=mul, rng=rng)
    print 'created grids'

    # create hash matrices
    hash_matrices = []
    for i in range(n_hashes):
        hash_matrices += [get_hash_matrix(t, d, rng=rng)]
    print 'created hash matrices'

    # hash pts into t dimension
    hashed_values = []
    for i in range(n_hashes):
        hashed_values += [np.dot(hash_matrices[i], pts.transpose()).transpose()]
    print 'hashed to t dimension'

    print 'finding plsh'
    # verbose interval: int(n/100)
    verbose_int = int(n/verbose_denom)
    
    # find plsh of pts
    full_hashes = []
    for i in range(n):
        hashed_pts = []
        for j in range(n_hashes):
            hashed_pts += [hashed_values[j][i]]
        full_hashes += [multi_hash(hashed_pts, shifts, w)]
        if i%verbose_int == 0 and verbose:
            print str(i), 'th pt done - ', full_hashes[i]
    print 'plsh hasing done'
    # first level hash - plsh
    full_hashes = np.array(full_hashes)
    return full_hashes

def second_hash(first_hash, denom):
    """hashes the first string with sha1 into a fixed range [denom]"""
    return int(int(hashlib.sha1(first_hash).hexdigest(), 16) % int(denom))


# Evaluation
def n_hashes_per_cluster(clust_ids, full_hashes):
    clusters = np.unique(clust_ids)
    n_hashes_counts = []
    for i in range(clusters.shape[0]):
        n_hashes_counts += [np.unique(full_hashes[np.where(clust_ids==clusters[i])[0]]).shape[0]]
    return clusters, n_hashes_counts

def clusters_in_buckets(clust_ids, full_hashes):
    buckets = np.unique(full_hashes)
    clusters = []
    for i in range(buckets.shape[0]):
        clusters += [clust_ids[np.where(full_hashes==buckets[i])[0]]]
    return buckets, clusters


def run_plsh_l1_eval(pts, clust_ids, t, w, n_hashes, mul=3, verbose=False, verbose_denom=10, rng=np.random.RandomState(42)):
    n = pts.shape[0]
    d = pts.shape[1]
    
    #create grids
    shifts = generate_grids(t, multiplier=mul, rng=rng)
    print 'created grids'

    # create hash matrices
    hash_matrices = []
    for i in range(n_hashes):
        hash_matrices += [get_hash_matrix(t, d, rng=rng)]
    print 'created hash matrices'

    # hash pts into t dimension
    hashed_values = []
    for i in range(n_hashes):
        hashed_values += [np.dot(hash_matrices[i], pts.transpose()).transpose()]
    print 'hashed to t dimension'

    print 'finding plsh'
    # verbose interval: int(n/100)
    verbose_int = int(n/verbose_denom)
    
    # find plsh of pts
    full_hashes = {}
    clusters = {}
    for i in range(n):
        hashed_pts = []
        for j in range(n_hashes):
            hashed_pts += [hashed_values[j][i]]
        multi_hashed = multi_hash(hashed_pts, shifts, w)
        # for # of clusters per bucket
        if multi_hashed not in full_hashes.keys():
            set_ = set()
            set_.add(clust_ids[i])
            full_hashes[multi_hashed] = set_
        else:
            set_ = full_hashes[multi_hashed]
            set_.add(clust_ids[i])
            full_hashes[multi_hashed] = set_   
        # for # of buckets per cluster
        if clust_ids[i] not in clusters.keys():
            set_ = set()
            set_.add(multi_hashed)
            clusters[clust_ids[i]] = set_
        else:
            set_ = clusters[clust_ids[i]]
            set_.add(multi_hashed)
            clusters[clust_ids[i]] = set_ 
        if i%verbose_int == 0 and verbose:
            print str(i), 'th pt done - '
    print 'plsh hasing done'
    # first level hash - plsh
    return full_hashes, clusters

def find_kmeans_obj(pts, centers):
    pw_dists = pairwise_distances(pts, centers)
    min_dists = np.min(pw_dists, axis=1)
    return np.sum(np.square(min_dists))