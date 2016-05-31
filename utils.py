import numpy as np

def random_ranking(N):
    ranking = np.empty((N, N), dtype=int)
    for i in range(N):
        ranking[i] = np.random.permutation(N).astype(int)
    return ranking


def random_bucket(sigma_A, sigma_B, proportion):
    a = np.random.binomial(1, p=proportion, size=sigma_A.shape[0]).astype(bool)
    return sigma_A * a[:, None] + sigma_B * np.logical_not(a[:, None]), a


def random_bucketing(n_nodes, proportion):
    return np.random.binomial(1, p=proportion, size=n_nodes).astype(int)


def test_similarity(sigma_1, sigma_2, max_value):
    nbr_similar_edges = np.sum(sigma_1[:max_value] - sigma_2[:max_value] == 0)
    nbr_diff_edges = np.product(sigma_1[:max_value].shape) - nbr_similar_edges
    return nbr_similar_edges, nbr_diff_edges


def bucket_counts(assignment, edges):
    N, N_A = len(assignment), sum(assignment)
    N_B = N - N_A
    vector = np.zeros(4, dtype=int)
    for (i, j) in edges:
        vector[2 * assignment[i] + assignment[j]] += 1
    BB, BA, AB, AA = vector
    return N_A, N_B, AA, AB, BA, BB


def naive_estimator(N_A, N_B, AA, AB, BA, BB):
    return (N_A + N_B)/N_A * (AA + AB)


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def bucketed_precision_rule_fixed(assignment, record):
    bucket_A_recs = record[assignment == 1]
    pos_counts = bucket_A_recs > 0
    neg_counts = bucket_A_recs < 0
    beta_params_pos = running_mean(np.sum(pos_counts, axis=0), 2)
    beta_params_neg = running_mean(np.sum(neg_counts, axis=0), 2)
    PBIT  = beta_params_pos / (beta_params_pos + beta_params_neg + 1e-8)
    def rule(i, j, rank):
        return np.random.rand() < PBIT[rank]
    return rule

def bucketed_precision_rule_nonfixed(assignment, record):
    bucket_A_recs = record[assignment == 1]
    pos_counts = bucket_A_recs >= 0
    neg_counts = bucket_A_recs < 0
    beta_params_pos = running_mean(np.sum(pos_counts, axis=0), 1)
    beta_params_neg = running_mean(np.sum(neg_counts, axis=0), 1)
    PBIT  = beta_params_pos / (beta_params_pos + beta_params_neg + 1e-8)
    def rule(i, j, rank):
        return np.random.rand() < PBIT[rank]
    return rule

def conf_interval(array):
    array = np.sort(array)
    l = array.shape[1]
    return array[:, int(l * 25/100)], array[:, int(l * 75/100)]

