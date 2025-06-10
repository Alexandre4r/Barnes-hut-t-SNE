import bhsne.tree as tree
import sys
# epsilon pour éviter les divisions par zéro
EPSILON = sys.float_info.epsilon


def bh_gradient(ea, pij, ydata, y_neighbors, theta2, n, paralel=False, pool=None):
    if paralel:
        t = tree.create_tree_parallel(ydata, n, pool)
    else:
        t = tree.create_tree(ydata, n)

    f_rep, Z = compute_f_rep(ydata, theta2, t, n)
    Z_total = sum(Z) + EPSILON

    f_att = compute_f_att(ea, pij, ydata, y_neighbors, n)

    grad = []
    for i in range(n):
        grad_row = []
        val = 4 * (f_att[i][0] - f_rep[i][0] / Z_total)
        grad_row.append(val)
        val = 4 * (f_att[i][1] - f_rep[i][1] / Z_total)
        grad_row.append(val)
        grad.append(grad_row)

    return grad  #, fin, debut


def is_far(node, sq_dist, theta2):
    # sq_dist != 0
    if sq_dist < EPSILON:
        return False
    if node.L2 / sq_dist < theta2:
        return True
    return False


def recurse_frep_Z(node, i, ydata, theta2, f_rep, Z):
    """
    Fonction récursive pour calculer Z et Z * frep pour un point i via Barnes-Hut.
    """
    if node.nb_body == 0:
        return

    diff_x = ydata[i][0] - node.com[0]
    diff_y = ydata[i][1] - node.com[1]
    dist2 = diff_x * diff_x + diff_y * diff_y

    #Cas approximation
    if is_far(node, dist2, theta2):
        nb = node.nb_body
        q = 1.0 / (1.0 + dist2)
        q2 = q * q
        f_rep[i][0] += nb * q2 * diff_x
        f_rep[i][1] += nb * q2 * diff_y
        Z[i] += nb * q
        return

    #Cas feuille
    if node.isLeaf:
        if node.body is not None and (
                abs(node.body[0] - ydata[i][0]) > EPSILON or abs(node.body[1] - ydata[i][1]) > EPSILON):
            diff_leaf_x = ydata[i][0] - node.body[0]
            diff_leaf_y = ydata[i][1] - node.body[1]
            dist2_leaf = diff_leaf_x * diff_leaf_x + diff_leaf_y * diff_leaf_y
            q = 1.0 / (1.0 + dist2_leaf)
            q2 = q * q
            f_rep[i][0] += q2 * diff_leaf_x * node.nb_body
            f_rep[i][1] += q2 * diff_leaf_y * node.nb_body
            Z[i] += q * node.nb_body
        return
    #Cas de base
    if node.child:
        for child in node.child:
            recurse_frep_Z(child, i, ydata, theta2, f_rep, Z)


def compute_f_rep(ydata, theta2, t, n):
    """
    Calcule Z et Z * forces répulsives via Barnes-Hut pour chaque point.
    """
    f_rep = [[0.0, 0.0] for _ in range(n)]
    Z = [0.0 for _ in range(n)]

    for i in range(n):
        recurse_frep_Z(t.node, i, ydata, theta2, f_rep, Z)

    return f_rep, Z


def compute_f_att(ea, pij, ydata, y_neighbors, n):
    f_att = [[0.0, 0.0] for _ in range(n)]
    for i in range(n):
        for j in y_neighbors[i]:
            diff_x = ydata[i][0] - ydata[j][0]
            diff_y = ydata[i][1] - ydata[j][1]
            norm2 = diff_x ** 2 + diff_y ** 2
            qijZ = 1.0 / (1.0 + norm2)
            f_att[i][0] += ea * pij[i][j] * qijZ * diff_x
            f_att[i][1] += ea * pij[i][j] * qijZ * diff_y
    return f_att
