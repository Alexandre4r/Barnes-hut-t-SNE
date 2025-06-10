import numpy as np

EPS = 1e-3

class Node:

    def __init__(self, x, y, L):
        self.com = None  # centre de masse
        self.body = None  # body dans la cellule
        self.isLeaf = True
        self.x = x
        self.y = y
        self.L = L  # taille
        self.L2 = L
        self.child = None  # tableau des nodes fils
        self.nb_body = 0

    def subdivide(self):
        """
        Ajoute quatres fils a l'arbre
        """
        new_L = (self.L) / 2
        self.child = [
            Node(self.x, self.y, new_L),
            Node(self.x + new_L, self.y, new_L),
            Node(self.x, self.y + new_L, new_L),
            Node(self.x + new_L, self.y + new_L, new_L)
        ]
        self.isLeaf = False
        return

    def _is_duplicate(self, point1: np.array, point2: np.array):

        return abs(point1[0] - point2[0]) < EPS and abs(point1[1] - point2[1]) < EPS

    def find(self, body):
        t = self.L / 2
        if body[1] > self.y + t:
            if body[0] > self.x + t:
                return 3
            else:
                return 2
        else:
            if body[0] > self.x + t:
                return 1
            else:
                return 0

    def insert(self, body):

        # Cas arbre vide
        if self.nb_body == 0:
            self.body = body
            self.L2 = np.square(self.L)

        # cas limite(taille de l'arbre)
        elif self.L < EPS:
            pass
        else:
            if self.body is not None:
                self.subdivide()
                self.child[self.find(self.body)].insert(self.body)
                self.body = None
            self.child[self.find(body)].insert(body)

        self.nb_body += 1
        return

    def center_of_mass(self):
        if self.body is not None:
            self.com = self.body
        elif self.child is not None:
            sum_x, sum_y, total = 0.0, 0.0, 0
            for child in self.child:
                if child is not None and child.nb_body != 0:
                    if child.com is None:
                        child.center_of_mass()
                    sum_x += child.com[0] * child.nb_body
                    sum_y += child.com[1] * child.nb_body
                    total += child.nb_body
            if total > 0:
                self.com = (sum_x / total, sum_y / total)
            else:
                self.com = None
        else:
            self.com = None
        return