import pygame
from pygame import *
#from random import randint
import sys
import numpy as np

def dat_to_numpy_elev(filename):
    with open(filename,'r') as f:
        raw_text = f.read()
    lst_2d = [row.strip().split() for row in raw_text.strip().split('\n')]
    elev = np.array(lst_2d, dtype=np.float32)
    return elev

def elev_to_grayscale(elev):
    arr_max = np.max(elev)
    arr_min = np.min(elev)
    color_arr = (255*(elev-arr_min)/(arr_max-arr_min)).astype(int)
    stack = np.stack((color_arr,color_arr,color_arr), axis=-1)
    return stack

def draw_map(gray, px=2):
    pygame.init()
    win = pygame.display.set_mode((len(gray[0])*px, len(gray)*px))  # based on map size
    print(f'DEBUG dim={gray.shape}')
    for r in range(len(gray)):
        for c in range(len(gray[r])):
            pygame.draw.rect(win, gray[r][c], (c*px, r*px, px, px))
    pygame.display.update()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
        #pygame.display.flip()

if __name__ == '__main__':
    elev = dat_to_numpy_elev('data/testMountains.dat')
    gray = elev_to_grayscale(elev)
    draw_map(gray,px=1)