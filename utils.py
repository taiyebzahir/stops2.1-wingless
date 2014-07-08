__author__ = 'Kamil Koziara & Taiyeb Zahir'

import numpy
import matplotlib.pyplot as plt
import matplotlib

def euclid_dist(a, b):
    ax, ay = a
    bx, by = b
    return numpy.sqrt(numpy.power(bx - ax, 2) + numpy.power(by - ay, 2))

def generate_pop(base_list):
    ll = []
    for k, l in base_list:
        ll += [l] * k
    return numpy.array(ll)


class HexGrid(object):
    """
    Class representing hexagonal grid.
    Attributes:
     - xy - centres of hexagons
     - shape - size
     - adj_mat - matrix with distances between hexagons
    """

    def __init__(self, n, m, d, dfun = euclid_dist):
        """
        Init method.
        Parameters:
         - n, m - size of the grid
         - d - euclid distance between centres of hexagons
         - dfun - method calculating distances for adjacency matrix
        """
        xoff = d * 1.
        yoff = d * numpy.cos(numpy.pi / 6.)

        ctr = []
        for i in range(n):
            offset = d / 2. if i % 2 else 0.
            for j in range(m):
                ctr.append( (xoff * j + offset, yoff * i))

        grid_size = n * m
        adj_mat = numpy.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(i + 1, grid_size, 1):
                adj_mat[i, j] = dfun(ctr[i], ctr[j])

        self.xy = ctr
        self.shape = [n, m]
        self.adj_mat = adj_mat + adj_mat.T


def unzip(l):
    return [list(t) for t in zip(*l)]

def draw_hex_grid(file, pop, grid, color_fun):
    """
    Simple method for drawing HexGrid using matplotlib.
    """
    q=1
    C = [0.1] * pop.shape[0]
    n,m = grid.shape
    y = []
    x = []
    for i in range(n):
        for j in range(m):
            y.append(q)
        q+=0.866
        if i % 2:
            x += list(numpy.linspace(1,m,m) + 0.5)
        else:
            x += list(numpy.linspace(1,m,m))
    for i, row in enumerate(pop):
        C[i] = color_fun(row)
    nor_m = matplotlib.colors.Normalize(vmin = 0, vmax = 1)
    plt.hexbin(numpy.array(x),numpy.array(y),numpy.array(C), gridsize=m+3, cmap=matplotlib.cm.rainbow, norm=nor_m, edgecolors= 'k', extent=[0, m +2,0, n +2])
    plt.draw()
    plt.savefig(file)
