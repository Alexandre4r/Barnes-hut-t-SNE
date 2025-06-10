from bhsne.node import Node

class Tree:
    def __init__(self, b_array, n):
        self.b_array = b_array
        self.n = n
        self.node = None
        res = find_root(b_array)
        if res is not None:
            self.x, self.y, self.L = res
        else:
            self.x = self.y = self.L = None

def find_root(b_array):
    # b_array = liste de listes [[x, y], ...]
    if len(b_array) <= 1:
        return None
    xmin = min(row[0] for row in b_array)
    ymin = min(row[1] for row in b_array)
    xmax = max(row[0] for row in b_array)
    ymax = max(row[1] for row in b_array)
    L = max(xmax - xmin, ymax - ymin)
    return xmin, ymin, L


def split_quadrants(b_array, x0, y0, L):
    '''
    Coupe le carré en 4 quadrants à partir de (x0, y0)
    '''
    half = L / 2
    quads = [[] for _ in range(4)]
    for x, y in b_array:
        if x < x0 + half and y < y0 + half:
            quads[0].append([x, y])
        elif x >= x0 + half and y < y0 + half:
            quads[1].append([x, y])
        elif x < x0 + half and y >= y0 + half:
            quads[2].append([x, y])
        else:
            quads[3].append([x, y])
    return quads

def create_tree(b_array, n):
    if b_array is None:
        return None
    t = Tree(b_array, n)
    t.node = Node(t.x, t.y, t.L)
    for i in range(len(b_array)):
        t.node.insert(b_array[i])
    t.node.center_of_mass()
    return t

def create_node_for_quadrant(args):
    q_points, x, y, L = args
    if not q_points:
        return None
    node = Node(x, y, L)
    for pt in q_points:
        node.insert(pt)
    node.center_of_mass()
    return node

def create_tree_parallel(b_array, n, pool):
    res = find_root(b_array)
    if res is None:
        return Tree(b_array, n)
    x0, y0, L = res
    quadrants = split_quadrants(b_array, x0, y0, L)
    half = L / 2

    # Coordonnées racines pour chaque quadrant
    quadrant_roots = [
        (quadrants[0], x0, y0, half),
        (quadrants[1], x0 + half, y0, half),
        (quadrants[2], x0, y0 + half, half),
        (quadrants[3], x0 + half, y0 + half, half)
    ]

    # Seulements quadrants  non vides
    args = [qr for qr in quadrant_roots if len(qr[0]) > 0]
    # Parallélise la création des nodes de quadrant
    nodes = pool.map(create_node_for_quadrant, args)

    t = Tree(b_array, n)
    if t.x is not None and t.y is not None and t.L is not None:
        t.node = Node(x0, y0, L)
        t.node.children = []
        q_iter = iter(nodes)
        for q in quadrants:
            if len(q) > 0:
                t.node.children.append(next(q_iter))
            else:
                t.node.children.append(None)

    return t