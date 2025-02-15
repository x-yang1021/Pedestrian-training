import numpy as np
import pandas as pd

def get_transparency(y, transparencies):
    for transparency in transparencies:
        if y > transparency[0] and y < transparency[1]:
            return 1
    return 0

def get_green_distance(x, green):
    return green[0] - x

def get_wall_distance(x, wall):
    return wall[0][0] - x