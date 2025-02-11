import numpy as np
import pandas as pd
import glob

North_wall = [(-474,52468),(-474,52322)]
North_green = [(-455,52468),(-455,52322)]
North_transparent = [(52407,52401),(52344,52337)]

North = True

# Load the data
if North:
    path = './Data/Xinjiekou/North'
    wall = North_wall
    green = North_green
    transparencies = North_transparent
all_files = glob.glob(path + "/*.txt")

