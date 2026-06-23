import time
import math
import numpy as np
from barnes_hut import bh_gradient
import multiprocessing

def grid_search(diff_i, i, perplexity):
    result = np.inf
    norm = np.linalg.norm(diff_i, axis=1)
    std_norm = np.std(norm)

    for sigma_search in np.linspace(0.01 * std_norm, 5 * std_norm, 200):
        p = np.exp(-(norm ** 2) / (2 * sigma_search ** 2))
        p[i] = 0
        epsilon = np.nextafter(0, 1)
        p_new = np.maximum(p / np.sum(p), epsilon)
        H = -np.sum(p_new * np.log2(p_new))
        if np.abs(np.log(perplexity) - H * np.log(2)) < np.abs(result):
            result = np.log(perplexity) - H * np.log(2)
            sigma = sigma_search
    return sigma

def condi_prob(xdata, perplexity):
    n = len(xdata)
    c_pij = np.zeros((n, n))
    for i in range(n):
        dist = xdata[i] - xdata
        norm = np.linalg.norm(dist, axis=1)
        sigma_i = grid_search(dist, i, perplexity)
        c_pij[i, :] = np.exp(-(norm ** 2) / (2 * sigma_i ** 2))

        np.fill_diagonal(c_pij, 0)
        c_pij[i, :] = c_pij[i, :] / np.sum(c_pij[i, :])
    epsilon = np.nextafter(0, 1)
    c_pij = np.maximum(c_pij, epsilon)
    return c_pij

def p_prob(xdata, perplexity):
    n = len(xdata)
    c_pij = condi_prob(xdata, perplexity)
    pij = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            pij[i, j] = (c_pij[i, j] + c_pij[j, i]) / (2 * n)
            pij[j, i] = pij[i, j]
    epsilon = np.nextafter(0, 1)
    pij = np.maximum(pij, epsilon)
    return pij

def set_y(xdata, d=2):
    return (np.random.normal(size=(len(xdata), d))).tolist()
'''
def q_prob(ydata):
    n = len(ydata)
    qij = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist2 = sum((ydata[i][k] - ydata[j][k]) ** 2 for k in range(len(ydata[i])))
            qij[i][j] = 1.0 / (1.0 + dist2)
    total = sum(qij[i][j] for i in range(n) for j in range(n) if i != j)
    epsilon = np.nextafter(0, 1)
    for i in range(n):
        for j in range(n):
            if i != j:
                qij[i][j] = max(qij[i][j] / total, epsilon)
            else:
                qij[i][j] = 0.0
    return qij
'''

def q_prob(ydata):
    ydata = np.array(ydata)
    n=len(ydata)
    qij = np.zeros((n,n))
    for i in range(n):
        dist = ydata[i] - ydata
        norm = np.linalg.norm(dist, axis = 1)
        qij[i,:] = (1 + norm**2) ** (-1)

    np.fill_diagonal(qij, 0)

    qij = qij / qij.sum()

    epsilon = np.nextafter(0,1)
    qij = np.maximum(qij, epsilon)

    return qij.tolist()

def get_y_neighbors(pij, perplexity):
    pij = np.array(pij)
    n_points = pij.shape[0]
    K = min(int(3 * perplexity), n_points - 1)
    pij = pij.copy()
    np.fill_diagonal(pij, -np.inf)
    neighbors = np.argpartition(-pij, K, axis=1)[:, :K]
    neighbors_sorted = np.take_along_axis(
        neighbors,
        np.argsort(-pij[np.arange(n_points)[:, None], neighbors], axis=1),
        axis=1
    )
    return neighbors_sorted.tolist()

def gradient(pij, qij, ydata):
    n = len(pij)
    d = len(ydata[0])
    grad = [[0.0 for _ in range(d)] for _ in range(n)]
    for i in range(n):
        dists = [[ydata[i][k] - ydata[j][k] for k in range(d)] for j in range(n)]
        norm_sq = [sum(dists[j][k] ** 2 for k in range(d)) for j in range(n)]
        g1 = [pij[i][j] - qij[i][j] for j in range(n)]
        g2 = [1.0 / (1.0 + norm_sq[j]) for j in range(n)]
        for k in range(d):
            s = 0.0
            for j in range(n):
                s += g1[j] * g2[j] * dists[j][k]
            grad[i][k] = 4.0 * s
    return grad

def tsne(xdata, perplexity, max_iter, step, d, method="naive", theta=0.2, par=False):
    theta2 = theta ** 2
    early_exaggeration = 8
    n = len(xdata)
    npxdata = np.array(xdata)
    pij = p_prob(npxdata, perplexity)
    y_neighbors = get_y_neighbors(pij, perplexity)

    Y = [[[0.0 for _ in range(d)] for _ in range(n)] for _ in range(max_iter)]
    Y0 = [[0.0 for _ in range(d)] for _ in range(n)]
    Y[0] = Y0
    Y1 = set_y(xdata, d)
    Y[1] = Y1

    pool = None
    if par:
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    i = 1
    alpha = 0.5

    list_cost = np.zeros(round(max_iter / 50))
    j=0
    while i < max_iter - 1:
        if i >= 500:
            alpha = 0.8
            early_exaggeration = 1

        if method == "naive":
            qij = q_prob(Y[i])
            grad = gradient([[early_exaggeration * pij[a][b] for b in range(n)] for a in range(n)], qij, Y[i])

        elif method == "bh":
            grad = bh_gradient(early_exaggeration, pij, Y[i], y_neighbors, theta2, n, par, pool)
        else:
            raise ValueError("method must be 'bh' or 'exact'")
        # Update Y[i+1] = Y[i] - step*grad + alpha*(Y[i] - Y[i-1])

        Y_next = []
        for a in range(n):
            row = []
            val = Y[i][a][0] - step * grad[a][0] + alpha * (Y[i][a][0] - Y[i - 1][a][0])
            row.append(val)
            val = Y[i][a][1] - step * grad[a][1] + alpha * (Y[i][a][1] - Y[i - 1][a][1])
            row.append(val)
            Y_next.append(row)

        Y[i+1] = Y_next

        if i % 50 == 0 or i == 1:
            epsilon = 1e-12
            qij = q_prob(Y[i])
            pij_np = np.array(pij)
            qij_np = np.array(qij)
            mask = ~np.eye(n, dtype=bool)
            pij_flat = pij_np[mask]
            qij_flat = qij_np[mask]
            cost = np.sum(pij_flat * np.log((pij_flat + epsilon) / (qij_flat + epsilon)))
            print(f"Iteration {i}: Value of Cost Function is {cost}")
            list_cost[j] = cost
            j += 1

        i += 1
    if par:
        if pool is not None:
            pool.close()
            pool.join()
    return Y[-1], Y, list_cost
