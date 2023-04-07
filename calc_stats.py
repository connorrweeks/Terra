from terrain_maker import TerrainMaker, seed
from civ_map import CivMap
from map_drawer import MapDrawer
from PIL import ImageTk, Image
from tkinter import *
import tkinter.messagebox as msgbox
import time
import numpy as np

h = 400
w = 400

biome_names = [
    'shallow',
    'deep',
    'alpine',
    'jungle',
    'forest',
    'grassland',
    'taiga',
    'desert',
    'tundra',
]

arr = np.zeros(9)

import pandas as pd

for i in range(30):
    tm = TerrainMaker()
    tm.load_pickle(i)

    b_prime = np.argmax(tm.biome_map, axis=2)
    unique, counts = np.unique(b_prime, return_counts=True)

    print(unique)
    print(counts)
    arr += counts

    rainfall = tm.rainfall.flatten()
    temp = tm.temperature.flatten()
    water = tm.water.flatten()

    df = pd.DataFrame()
    df["rainfall"] = rainfall
    df["temp"] = temp
    df["water"] = water
    df.to_csv(f"./stats/{i}_rainfall_temp.csv", index=False)

for i in range(2, 9):
    print(f"{biome_names[i]} - {(arr[i] / sum(arr[2:])):.2%}")