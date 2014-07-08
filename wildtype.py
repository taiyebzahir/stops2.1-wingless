__author__ = 'Kamil Koziara & Taiyeb Zahir'

import cProfile
import numpy
from utils import generate_pop, HexGrid, draw_hex_grid
from stops_ import Stops2

secretion = numpy.array([3])
reception = numpy.array([2])
receptors = numpy.array([-1])
bound=numpy.array([1,1,1,1,1])

base1=numpy.array([1,0,0,0,1])
base2=numpy.array([0,0,0,0,1])

trans_mat= numpy.matrix(numpy.array([\
                       [0,5,0,0,0], #g0
                       [0,0,2,1,0], #g1
                       [0,0,0,0,0], #g2
                       [0,0,0,0,0], #g3
                       [0,0,-1,0,0] #g4
                       ]))

init_pop = generate_pop([(1150, base2), (200, base1), (1150, base2)])
grid = HexGrid(50, 50, 1)

def color_fun(row):
    if row[2]==1:
        return 0
    else:
        return 1.



def run():
    x = Stops2(trans_mat, init_pop, grid.adj_mat, bound, secretion, reception, receptors, secr_amount=1, leak=0, max_con=6, max_dist=15, opencl=False)
    for i in range(200):
        x.step()
        if i%5 == 0:
            print i
            draw_hex_grid("pics/wildtype%04d.png"%i, x.pop, grid, color_fun)


cProfile.run("run()")
