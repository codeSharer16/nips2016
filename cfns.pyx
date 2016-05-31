import utils
import main
cimport numpy as np
import numpy as np
import networkx as nx
cimport cython
DTYPE = np.int
ctypedef np.int_t DTYPE_t

# --- strongly fixed ---

def process_strongly_fixed(np.ndarray[DTYPE_t, ndim=2] sigma, int T, true_edges,
                           int seed):
    cdef int N = len(sigma)
    cdef np.ndarray[DTYPE_t, ndim=1] indices = np.zeros(N, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] record = np.zeros((N, N), dtype=DTYPE)
    pos_edges, neg_edges = set(), set()
    cdef int t, i, j, k
    rng = np.random.RandomState(seed)
    for t in range(T):
        for i in rng.permutation(N):
            j = sigma[i, indices[i]]
            while (i, j) in pos_edges or (j, i) in pos_edges:
                record[i, indices[i]] = 0  # rec is skipped
                indices[i] += 1
                j = sigma[i, indices[i]]
            if (i, j) in true_edges:
                pos_edges.add((i,j))
                record[i, indices[i]] = 1  # valid rec
            else:
                neg_edges.add((i, j))
                record[i, indices[i]] = -1  # invalid rec
            indices[i] += 1
    return pos_edges, neg_edges, record


def simul_process_strongly_fixed(np.ndarray[DTYPE_t, ndim=2] sigma, int T, rule,
                                 pos_edges, neg_edges, int seed):
    cdef int N = len(sigma)
    cdef np.ndarray[DTYPE_t] indices = np.zeros(N, dtype=DTYPE)
    cdef int t, i, j, k
    indices = np.zeros(N, dtype=int)
    guessed_edges = set()
    rng = np.random.RandomState(seed)
    for t in range(T):
        for i in rng.permutation(N):
            j = sigma[i, indices[i]]
            while (i, j) in guessed_edges or (j, i) in guessed_edges:
                indices[i] += 1
                j = sigma[i, indices[i]]
            if (i,j) in pos_edges or (j,i) in pos_edges or ( (i,j) not in
                    neg_edges and (j,i) not in neg_edges and rule(i, j,
                        indices[i])):
                guessed_edges.add((i, j))
            indices[i] += 1 # min(indices[i] + 1, N -1)
    return guessed_edges


def run_strongly_fixed(T, true_edges, treatment, control, proportion, n_iter=200,
                        seed=100):
    mdi_est = np.zeros(n_iter)
    naive_est = np.zeros(n_iter)
    for i in range(n_iter):
        sigma, assignment = utils.random_bucket(treatment, control, proportion)
        pos_edges, neg_edges, record = process_strongly_fixed(sigma, T,
                                                              true_edges, seed)
        naive_est[i] = utils.naive_estimator(*utils.bucket_counts(assignment,
                                       pos_edges))
        prec_rule = utils.bucketed_precision_rule_fixed(assignment, record)
        mdi_edges = simul_process_strongly_fixed(treatment, T, prec_rule,
                pos_edges, neg_edges, seed)
        mdi_est[i] = utils.naive_estimator(*utils.bucket_counts(assignment,
                                                                mdi_edges))
    return naive_est, mdi_est

# --- weakly fixed ---

def process_weakly_fixed(np.ndarray[DTYPE_t, ndim=2] sigma, int T, true_edges,
                         int seed):
    cdef int N = len(sigma)
    cdef np.ndarray[DTYPE_t, ndim=1] indices = np.zeros(N, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] record = np.zeros((N, N), dtype=DTYPE)
    pos_edges, neg_edges = set(), set()
    cdef int t, i, j, k
    rng = np.random.RandomState(seed)
    for t in range(T):
        for i in rng.permutation(N):
            j = sigma[i, indices[i]]
            while (j,i) in pos_edges or (j,i) in neg_edges:
                #is_pos = int((j,i) in pos_edges)
                record[i, indices[i]] = 0 #2*is_pos-1   # rec is skipped
                indices[i] += 1
                j = sigma[i, indices[i]]
            if (i, j) in true_edges:
                pos_edges.add((i,j))
                record[i, indices[i]] = 1  # valid rec
            else:
                neg_edges.add((i, j))
                record[i, indices[i]] = -1  # invalid rec
            indices[i] += 1
    return pos_edges, neg_edges, record


def simul_process_weakly_fixed(np.ndarray[DTYPE_t, ndim=2] sigma, int T, rule,
                               pos_edges, neg_edges, int seed):
    cdef int N = len(sigma)
    cdef np.ndarray[DTYPE_t] indices = np.zeros(N, dtype=DTYPE)
    cdef int t, i, j, k
    indices = np.zeros(N, dtype=int)
    guessed_edges = set()
    neg_guessed_edges = set()
    rng = np.random.RandomState(seed)
    for t in range(T):
        for i in rng.permutation(N):
            j = sigma[i, indices[i]]
            while (j,i) in guessed_edges or (j,i) in neg_guessed_edges:
                indices[i] += 1
                j = sigma[i, indices[i]]
            if (i,j) in pos_edges or (j,i) in pos_edges or ((i,j) not in
                    neg_edges and (j,i) not in neg_edges and rule(i, j,
                        indices[i])):
                guessed_edges.add((i, j))
            else:
                neg_guessed_edges.add((i,j))
            indices[i] += 1
    return guessed_edges


def run_weakly_fixed(T, true_edges, treatment, control, proportion, n_iter=200,
                     seed=100):
    mdi_est = np.zeros(n_iter)
    naive_est = np.zeros(n_iter)
    for i in range(n_iter):
        sigma, assignment = utils.random_bucket(treatment, control, proportion)
        pos_edges, neg_edges, record = process_weakly_fixed(sigma, T,
                                                            true_edges, seed)
        naive_est[i] = utils.naive_estimator(*utils.bucket_counts(assignment,
                                             pos_edges))
        prec_rule = utils.bucketed_precision_rule_fixed(assignment, record)
        mdi_edges = simul_process_weakly_fixed(treatment, T, prec_rule,
                                                pos_edges, neg_edges, seed)
        mdi_est[i] = utils.naive_estimator(*utils.bucket_counts(assignment,
                                                                mdi_edges))
    return naive_est, mdi_est



# --- non-fixed ---

class PermutationAlgorithm():
    def __init__(self, sigma):
        self.recs = sigma
        self.indices = np.zeros(len(sigma), dtype=int)

    def __call__(self, user, obs_graph):
        index = self.indices[user]
        val = self.recs[user, index]
        while val in obs_graph.friend(user):
            index += 1
            val = self.recs[user, index]
        self.indices[user] = index + 1
        return val

    def reset(self):
        self.indices = np.zeros(len(self.recs))


class FoFandPermutationAlgorithm():
    def __init__(self, sigma, epsilon):
        self.recs = sigma
        self.indices = np.zeros(len(sigma), dtype=int)
        self.eps = epsilon  # proba with which to pick fof
        self.rng = np.random.RandomState(200)
        self.rng2 = np.random.RandomState(100)

    def __call__(self, user, obs_graph):
        do_random = 1
        if self.rng.rand() < self.eps:
            candidates = obs_graph.fofnotfriend(user)
            n_candidates = len(candidates)
            if n_candidates > 0:
                val = candidates[self.rng2.randint(n_candidates)]
                do_random = 0
        if do_random:
            index = self.indices[user]
            val = self.recs[user, index]
            while val in obs_graph.friend(user):
                index += 1
                val = self.recs[user, index]
            self.indices[user] = index + 1
        return val

    def reset(self):
        self.indices = np.zeros(len(self.recs), dtype=int)
        self.rng = np.random.RandomState(200)
        self.rng2 = np.random.RandomState(100)


def process_nonfixed(true_edges, treatment, control,
                    np.ndarray[DTYPE_t, ndim=1] buckets, int N, int T, int buff,
                    int seed):
    obs_graph = main.UndirectedGraph()
    obs_graph.add_nodes(range(N))
    cdef np.ndarray[DTYPE_t, ndim=2] potential_recs = np.zeros((N, T + buff),
        dtype=int)
    cdef np.ndarray[DTYPE_t, ndim=2] record = np.zeros( (N, T), dtype=int)
    cdef int t, bucket, user, val
    pos_edges, neg_edges = set(), set()
    rng = np.random.RandomState(seed)
    for t in range(T):
        for user in rng.permutation(range(N)):
            bucket = buckets[user]
            if bucket:
                val = treatment(user, obs_graph)
                potential_recs[user, t] = val
            else:
                val = control(user, obs_graph)
                potential_recs[user, t] = treatment(user, obs_graph)
            rec_edge = (user, val)
            if rec_edge in true_edges:
                obs_graph.add_edge(*rec_edge)
                pos_edges.add(rec_edge)
                record[user, t] = 1
            else:
                neg_edges.add(rec_edge)
                record[user, t] = -1
    for t in range(buff):  # store future recommendations for simul_process
        for user in range(N):
            potential_recs[user, t + T] = treatment(user, obs_graph)
    return pos_edges, neg_edges, record, potential_recs


def simul_process_partial(np.ndarray[DTYPE_t, ndim=2] potential_recs, rule,
                          pos_edges, neg_edges, int T, int N, int seed):
    assert np.max(potential_recs) < N
    guessed_edges = set()
    rng = np.random.RandomState(seed)
    cdef np.ndarray[DTYPE_t, ndim=1] indices = np.zeros(N, dtype=int)
    cdef int t, user, rec_user
    for t in range(T):
        for user in rng.permutation(range(N)):
            rec_user = potential_recs[user, indices[user]]
            while (user, rec_user) in guessed_edges or (rec_user, user) in \
                    guessed_edges:
                indices[user] += 1
                rec_user = potential_recs[user, indices[user]]
            if ((user, rec_user) in pos_edges or (rec_user, user) in pos_edges
                    or ((user, rec_user) not in neg_edges and (rec_user, user)
                        not in neg_edges and rule(user, rec_user, t))):
                guessed_edges.add((user, rec_user))
            indices[user] += 1
    return guessed_edges


def simul_process_full(treatment, rule, pos_edges, neg_edges, int T, int N,
                        int seed):
    guessed_edges = set()
    rng = np.random.RandomState(seed)
    cdef np.ndarray[DTYPE_t, ndim=1] indices = np.zeros(N, dtype=int)
    cdef int t, user, rec_user
    obs_graph = main.UndirectedGraph()
    obs_graph.add_nodes(range(N))
    for t in range(T):
        for user in rng.permutation(range(N)):
            rec_user = treatment(user, obs_graph)
            if ((user, rec_user) in pos_edges or (rec_user, user) in pos_edges
                    or ((user, rec_user) not in neg_edges and (rec_user, user)
                        not in neg_edges and rule(user, rec_user, t))):
                guessed_edges.add((user, rec_user))
                obs_graph.add_edge(user, rec_user)
    return guessed_edges


def run_nonfixed(T, N, true_edges, treatment, control, proportion, n_iter, buff,
                 seed):
    partial_mdi_est = np.zeros(n_iter)
    full_mdi_est = np.zeros(n_iter)
    naive_est = np.zeros(n_iter)
    for i in range(n_iter):
        buckets = utils.random_bucketing(N, proportion)
        pos_edges, neg_edges, record, pot_recs = process_nonfixed(
                                                         true_edges=true_edges,
                                                         treatment=treatment,
                                                         control=control,
                                                         buckets=buckets,
                                                         N=N, T=T, buff=buff,
                                                         seed=seed)
        naive_est[i] = utils.naive_estimator(*utils.bucket_counts(buckets,
                                                                  pos_edges))
        prec_rule = utils.bucketed_precision_rule_nonfixed(buckets, record)
        partial_mdi_edges = simul_process_partial(potential_recs=pot_recs,
                                                  rule=prec_rule,
                                                  pos_edges=pos_edges,
                                                  neg_edges=neg_edges,
                                                  T=T, N=N, seed=seed)
        partial_mdi_est[i] = utils.naive_estimator(*utils.bucket_counts(buckets,
                                                            partial_mdi_edges))
        full_mdi_edges = simul_process_full(treatment=treatment,
                                            rule=prec_rule,
                                            pos_edges=pos_edges,
                                            neg_edges=neg_edges,
                                            T=T, N=N, seed=seed)
        full_mdi_est[i] = utils.naive_estimator(*utils.bucket_counts(buckets,
                                                         full_mdi_edges))
        for alg in [treatment, control]:
            alg.reset()
    return naive_est, partial_mdi_est, full_mdi_est

# -- create baseline permutations with bias

@cython.boundscheck(False)
def make_stronger_algo(np.ndarray[DTYPE_t, ndim=2] sigma_1, int k,
                               int T, true_edges):
    cdef int N = len(sigma_1)
    cdef np.ndarray[DTYPE_t, ndim=2] sigma_2 = sigma_1.copy()
    cdef int i, j, val
    cdef float cutoff = .5
    cdef np.ndarray[DTYPE_t, ndim=2] rand_int = np.random.randint(N, size=(k,3))
    for i, j, k in rand_int:
        val = sigma_2[i][j]
        if (i, val) in true_edges:
            if j > T and k < T:
                sigma_2[i][j] = sigma_1[i][k]
                sigma_2[i][k] = sigma_1[i][j]
        elif np.random.rand() < cutoff and (i, sigma_2[i][k]) not in true_edges:
            sigma_2[i][j] = sigma_1[i][k]
            sigma_2[i][k] = sigma_1[i][j]
    return sigma_2

