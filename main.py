__author__ = 'cpolzak'

from search import *
from drawMap import *

elev = dat_to_numpy_elev('data/Colorado_480x480.dat')
gray = elev_to_grayscale(elev)

search_alg = SemiFringeSearch(elev, pixel_size=300, heuristic_lambda=1, elev_lambda=.8)
path, searched_nodes = search_alg.search(210, 310) #210, 350
for coord in searched_nodes:
    gray[coord][2] = 200
for coord in path:
    gray[coord] = (255,0,0)

draw_map(gray, px=1)
