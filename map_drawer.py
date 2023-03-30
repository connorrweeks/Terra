#from scene import *
#import scene
from vect2 import Vector2
import sound
import random as r
import math
import time
import numpy as np
from PIL import Image
import io
import os
import sys
import traceback
from perlin import SimplexNoise
import pickle
import shutil

ZW = 12
RW = 10
UW = 5

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

min_names = [
    'copper',
    'iron',
    'gold',
    'gems',
    'salt',
]
b_colors = [
    [70, 70, 210],  # shallow
    [50, 50, 96],  # deep
    [100, 100, 100],  # alpine
    [20, 160, 40],  # rainforest
    # [82, 110, 32],#rainforest
    # [0, 130, 60],#forest
    [38, 90, 37],  # forest
    # [80, 200, 120],#grasslands
    [107, 162, 87],  # grasslands
    # [0, 100, 30],#taiga
    [30, 119, 105],  # taiga
    [237, 201, 175],  # desert
    [230, 230, 230],  # tundra
    # [165, 171, 181],#tundra
]
b_colors = np.array(b_colors)


def softplus(x):
    return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x,0)

def softclip(x, a=0, b=1, c=5):
    v = x
    v = v + softplus(-c*(x)) / c
    v = v - softplus(c*(x - 1)) / c
    return v


class MapDrawer():
    def __init__(self, w, h, view="biome"):
        self.w = w
        self.h = h
        self.view = view

        self.points = None
        self.draw = None
        self.isle_colors = None
        self.prov_colors = None
        self.area_colors = None

        self.view_int = 0
        self.view_functions = {"land":self.view3, "biome":self.view1, "island":self.view2, "wind":self.view4, "area":self.area_view ,"province":self.province_view}

    def set_view_int(self, view_int):
        self.view_int = view_int
        self.view = list(self.view_functions.keys())[self.view_int]

    def set_view(self, view):
        self.view = view
        assert(view in ['land', 'biome', 'island', 'wind', 'civ'])

    def to_colors(self, tm):
        arr = tm.elevation
        v_slope = arr[1:, :] - arr[:-1, :]
        self.arr = arr

        v_slope = v_slope / np.max(v_slope)
        self.v_slope = np.concatenate([np.zeros((1, arr.shape[1])), v_slope])
        cm = self.view_functions[self.view](tm)
        return cm

    # draw =self.draw_vectors(tm.slope_vectors)
    # draw = tm.ocean.clip(0,1)

    # sun = step_map(tm.rainfall, 5)

    def view1(self, tm):
        t0 = time.perf_counter()
        # arr = step_map(self.arr, 4)
        arr = tm.elevation
        cm = np.zeros(list(arr.shape) + [3])
        for row in range(arr.shape[0]):
            b_prime = np.argmax(tm.biome_map[row], axis=1)
            row_colors = b_colors[b_prime.astype(np.uint8)]
            cm[row, :, :] = row_colors
        noise_mult = np.array([5, 5, 5, 20, 20, 5, 20, 5, 5])
        noise_mult = noise_mult[np.argmax(tm.biome_map, axis=2)]
        noise = tm.color_noise * noise_mult[:, :, None]
        #cm *= 0.85 * (1 + (self.v_slope[:, :, None] * 0.6))
        cm *= 1 * (1 + (self.v_slope[:, :, None] * 3))
        cm += noise
        cm = softclip(cm/255, c=10)*255


        print(f'view1: {time.perf_counter() - t0:.2f}')

        # special
        # cm = self.draw_deposits(tm.deposits, cm)

        return cm.clip(0, 255).astype(np.uint8)

    def view2(self, tm):
        t0 = time.perf_counter()

        if (self.isle_colors is None):
            self.isle_colors = np.random.randint(100, 255, (len(tm.island_sizes), 3)).astype(np.uint8)
        # print(isle_colors.shape)

        arr = tm.elevation
        cm = np.zeros(list(arr.shape) + [3])

        island_mask = (tm.island != -1).astype(np.uint8)
        island_colors = self.isle_colors[np.abs(tm.island).astype(np.uint8)]

        cm = self.view1(tm)
        cm = cm * (1 - island_mask)[:, :, None]
        cm += island_colors * island_mask[:, :, None]

        print(f'view2: {time.perf_counter() - t0:.2f}')

        #cm = self.draw_deposits(tm.deposits, cm)

        return cm.astype(np.uint8)


    def area_view(self, tm):
        t0 = time.perf_counter()

        if (self.area_colors is None):
            self.area_colors = np.random.randint(100, 255, (np.max(tm.areas)+1, 3)).astype(np.uint8)
        # print(isle_colors.shape)

        arr = tm.elevation
        cm = np.zeros(list(arr.shape) + [3])

        province_mask = (tm.areas != -1).astype(np.uint8)
        province_colors = self.area_colors[np.abs(tm.areas).astype(np.uint8)]

        cm = self.view1(tm)
        cm = cm * (1 - province_mask)[:, :, None]
        cm += province_colors * province_mask[:, :, None]

        print(f'view2: {time.perf_counter() - t0:.2f}')

        #cm = self.draw_deposits(tm.deposits, cm)

        return cm.astype(np.uint8)

    def province_view(self, tm):
        t0 = time.perf_counter()

        if (self.prov_colors is None):
            self.prov_colors = np.random.randint(100, 255, (np.max(tm.provinces)+1, 3)).astype(np.uint8)
        # print(isle_colors.shape)

        arr = tm.elevation
        cm = np.zeros(list(arr.shape) + [3])

        province_mask = (tm.provinces != -1).astype(np.uint8)
        province_colors = self.prov_colors[np.abs(tm.provinces).astype(np.uint8)]

        cm = self.view1(tm)
        cm = cm * (1 - province_mask)[:, :, None]
        cm += province_colors * province_mask[:, :, None]

        print(f'view2: {time.perf_counter() - t0:.2f}')

        #cm = self.draw_deposits(tm.deposits, cm)

        return cm.astype(np.uint8)

    def view3(self, tm):
        cm = np.array([50, 50, 160])[None, None, :] * tm.water[:, :, None]
        cm += np.array([0, 180, 60])[None, None, :] * (1 - tm.water[:, :, None])
        return cm.astype(np.uint8)

    def __view4(self, tm):
        arr = tm.run_off
        cm = self.view1(tm) * (1 - arr[:, :, None])
        cm += np.array([255, 255, 255])[None, None, :] * arr[:, :, None]
        return cm.astype(np.uint8)

    def view4(self, tm):
        arr = self.draw_vectors(tm.air_currents)
        cm = self.view1(tm) * (1 - arr[:, :, None])
        cm += np.array([255, 255, 255])[None, None, :] * arr[:, :, None]
        return cm.clip(0,255).astype(np.uint8)

    def view5(self, tm):  # rainfall
        dry = np.array([30, 30, 200])
        wet = np.array([200, 200, 30])
        cm = (tm.rainfall[:, :, None] * wet[None, None, :])
        cm += ((1 - tm.rainfall[:, :, None]) * dry[None, None, :])
        cm = cm * (1 - tm.water)[:, :, None]
        cm += (self.view1(tm) * tm.water[:, :, None])
        return cm.astype(np.uint8)

    def view6(self, tm):  # temp
        dry = np.array([30, 30, 200])
        wet = np.array([200, 200, 30])
        cm = (tm.temperature[:, :, None] * wet[None, None, :])
        cm += ((1 - tm.temperature[:, :, None]) * dry[None, None, :])
        cm = cm * (1 - tm.water)[:, :, None]
        cm += (self.view1(tm) * tm.water[:, :, None])
        return cm.astype(np.uint8)

    def view8(self, tm):  # normal
        arr = tm.elevation

        cm = self.view1(tm)
        cm = self.draw_deposits(tm.deposits, cm)
        civ_zones, zone_mask = self.get_civ_zones(tm)

        if (len(tm.all_civs) > 0):
            colored_zones = self.get_zone_colors(tm, zone_mask, civ_zones)

            border_mask = self.get_border_mask(civ_zones)

            cm = (cm * (1 - (border_mask[:, :, None] * zone_mask[:, :, None])))
            cm += colored_zones * border_mask[:, :, None] * zone_mask[:, :, None]
        # cm -= colored_zones * border_mask[:,:,None]

        urban_mask = self.get_urban_mask(tm)
        # urban_mask -= np.abs(tm.color_noise[:,:,0] * 0.05)
        c_color = 150 * np.array([1, 1, 1])
        cm = self.mix_mask(cm, urban_mask, c_color)

        rural_mask = self.get_rural_mask(tm) * 0.5
        c_color = 150 * np.array([0, 1, 0])
        cm = self.mix_mask(cm, rural_mask, c_color)

        cm = self.add_movers(cm, tm)

        return cm.astype(np.uint8)

    def view7(self, tm):  # borders
        cm = self.view1(tm)
        cm = self.draw_deposits(tm.deposits, cm)
        civ_zones, zone_mask = self.get_civ_zones(tm)

        if (len(tm.all_civs) > 0):
            colored_zones = self.get_zone_colors(tm, zone_mask, civ_zones)

            cm = (cm * (1 - zone_mask[:, :, None]))
            cm += colored_zones

        city_mask = self.get_city_mask(tm)
        c_color = np.array([0, 0, 0])
        cm = self.mix_mask(cm, city_mask, c_color)

        cm = self.add_movers(cm, tm)

        return cm.astype(np.uint8)

    def view9(self, tm):  # occupied
        cm = self.view1(tm)

        cm = self.draw_deposits(tm.deposits, cm)
        civ_zones, zone_mask = self.get_civ_zones(tm)

        if (len(tm.all_civs) > 0):
            colored_zones = self.get_zone_colors(tm, zone_mask, civ_zones)

            # b_mask = 1 - (self. get_border_mask(civ_zones) * 0.5)

            o_mask = self.get_o_mask(tm) * 0.5

            cm = (cm * (1 - zone_mask[:, :, None]))

            c = np.array([255, 0, 0])
            cm = self.mix_mask(cm, o_mask, c)

            cm += colored_zones * (1 - o_mask[:, :, None])

        cm = self.add_movers(cm, tm)

        return cm.astype(np.uint8)

    def add_movers(self, cm, tm):
        army_mask = self.get_army_mask(tm)
        a_color = np.array([255, 255, 255])
        cm = self.mix_mask(cm, army_mask, a_color)

        # trader_mask = self.get_trader_mask(tm)
        # t_color = np.array([255,255,0])
        # cm = self.mix_mask(cm, trader_mask, t_color)
        return cm

    def get_civ_zones(self, tm):
        civ_zones = np.full(tm.slope.shape, -1)
        zone_mask = np.zeros(tm.slope.shape)
        i = 0
        for c in tm.all_civs:
            i += 1
            civ_zones += c.zone * i
            zone_mask += c.zone
        return civ_zones, zone_mask

    def get_zone_colors(self, tm, zone_mask, civ_zones):
        civ_colors = np.vstack([x.color for x in tm.all_civs])
        colored_zones = civ_colors[civ_zones.clip(0, len(tm.all_civs) - 1).astype(np.int)] * zone_mask[:, :, None]
        return colored_zones

    def get_city_mask(self, tm):
        city_mask = np.zeros(tm.slope.shape)
        for c in tm.all_cities:
            if (c.level == 0):
                city_mask[c.y, c.x] = 1
            elif (c.level == 1):
                city_mask[c.y - 1:c.y + 2, c.x - 1:c.x + 2] = 1
            else:
                city_mask[c.y - 2:c.y + 3, c.x - 2:c.x + 3] = 1
        return city_mask

    def get_rural_mask(self, tm):
        rural_mask = np.zeros((tm.slope.shape[0] + ZW + ZW, tm.slope.shape[1] + ZW + ZW))
        for c in tm.all_cities:
            rural_mask[c.y:c.y + 1 + ZW * 2, c.x:c.x + 1 + ZW * 2] += c.Rzone
        return rural_mask[ZW:-ZW, ZW:-ZW]

    def get_urban_mask(self, tm):
        urban_mask = np.zeros((tm.slope.shape[0] + ZW + ZW, tm.slope.shape[1] + ZW + ZW))
        for c in tm.all_cities:
            # print(urban_mask[c.y:c.y+2+ZW*2, c.x:c.x+2+ZW*2].shape)
            # print(c.Uzone.shape)
            urban_mask[c.y:c.y + 1 + ZW * 2, c.x:c.x + 1 + ZW * 2] += c.Uzone
        return urban_mask[ZW:-ZW, ZW:-ZW]

    def get_o_mask(self, tm):
        o_mask = np.zeros((tm.water.shape[0] + ZW + ZW, tm.water.shape[1] + ZW + ZW))
        for c in tm.all_cities:
            o_mask[c.y:c.y + 1 + ZW * 2, c.x:c.x + 1 + ZW * 2] += c.zone * c.occupied
        return o_mask[ZW:-ZW, ZW:-ZW]

    def get_army_mask(self, tm):
        army_mask = np.zeros(tm.slope.shape)
        for a in tm.all_armies:
            army_mask[a.y, a.x] = 1
        return army_mask

    def get_trader_mask(self, tm):
        trader_mask = np.zeros(tm.slope.shape)
        for a in tm.all_cities:
            if (a.trader != None):
                trader_mask[a.trader.y, a.trader.x] = 1
        return trader_mask

    def mix_mask(self, cm, mask, c):
        return (cm * (1 - mask[:, :, None])) + ((mask[:, :, None]) * c[None, None, :])

    def get_border_mask(self, arr):
        border_mask = np.zeros(arr.shape)

        sum_array = np.zeros((arr.shape[0] - 2, arr.shape[1] - 2))

        y_len = arr.shape[0]
        x_len = arr.shape[1]
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                sum_array += arr[1 + y:y_len - 1 + y, 1 + x:x_len - 1 + x]
        border_mask[1:-1, 1:-1] = (sum_array != (arr[1:-1, 1:-1] * 9)).astype(np.uint8)

        return border_mask

    def to_image(self, cm):
        im = Image.fromarray(cm)
        return im

    def to_ui(self, im, scale=1):
        im = im.resize((int(self.w * scale), int(self.h * scale)))
        with io.BytesIO() as bIO:
            im.save(bIO, 'PNG')
            img = ui.Image.from_data(bIO.getvalue())
        return img

    def draw_deposits(self, deposits, cm):
        min_colors = [
            [183, 168, 70],  # copper
            [194, 194, 194],  # iron
            [253, 251, 102],  # gold
            [159, 106, 34],  # gems
            [250, 250, 250],  # salt
        ]
        for d in deposits:
            x, y, min = d
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if (abs(dx) + abs(dy) > 2):
                        continue
                    cm[y + dy][x + dx] = min_colors[min]
        return cm

    def draw_vectors(self, vectors):
        max_time = 15
        line_len = 5
        num_points = 1000

        if (self.points is None):
            self.points = np.zeros((num_points, 2))
            self.points[:, 0] = np.random.randint(0, vectors.shape[0], size=num_points)
            self.points[:, 1] = np.random.randint(0, vectors.shape[1], size=num_points)

            self.point_times = np.zeros(num_points)
            self.point_times = np.random.randint(0, max_time, size=num_points) + 1
        else:
            rpoints = np.zeros((num_points, 2))
            rpoints[:, 0] = np.random.randint(0, vectors.shape[0], size=num_points)
            rpoints[:, 1] = np.random.randint(0, vectors.shape[1], size=num_points)

            mask = (self.point_times >= max_time)
            self.points = (rpoints * mask[:,None]) + ((1 - mask)[:,None] * self.points)

            self.point_times = self.point_times % max_time

        if (self.draw is None):
            self.draw = np.zeros((vectors.shape[0], vectors.shape[1]))
        self.draw -= 1 / max_time
        self.draw = self.draw.clip(0, 1)

        for j in range(line_len):
            point_vecs = vectors[self.points[:,0].astype(int), self.points[:,1].astype(int), :]
            point_vecs = np.flip(point_vecs, axis=1)

            abs_vec = np.sqrt(np.sum(point_vecs * point_vecs, axis=1)) + 0.00001
            self.points = self.points.astype(np.float64) + (point_vecs / abs_vec[:, None])
            self.points[:, 0] = self.points[:, 0].clip(0, vectors.shape[0]-1)
            self.points[:, 1] = self.points[:, 1].clip(0, vectors.shape[1]-1)

            self.draw[self.points[:, 0].astype(np.int64), self.points[:, 1].astype(np.int64)] = ((j + 1) / line_len * max_time) + ((max_time - 1) / max_time)

        self.point_times = self.point_times + 1

        return self.draw

        for i in range(num_points):
            p, t = self.points[i]
            # draw[p.y, p.x] = 0
            for j in range(line_len
                           ):
                vec = Vector2(vectors[int(p.y), int(p.x), 0], vectors[int(p.y), int(p.x), 1])
                if (abs(vec) == 0):
                    break
                p += vec / abs(vec)
                if (p.x < 0 or p.y < 0 or p.x >= self.draw.shape[1] or p.y >= self.draw.shape[0]):
                    t = max_time - 1
                    break
                self.draw[int(p.y), int(p.x)] = min(
                    max(self.draw[int(p.y), int(p.x)], ((j + 1) / line_len * max_time) + ((max_time - 1) / max_time),
                        0), 1)
            #self.points[i] = (p, (t + 1))

        for i in range(num_points):
            self.points[i] = (p, (t + 1))
        # self.draw.clip(0,1)

        return self.draw

    def draw_vectors2(self, vectors):
        draw = np.zeros((vectors.shape[0], vectors.shape[1]))
        for y_ in range(4, vectors.shape[0], 5):
            for x_ in range(4, vectors.shape[1], 5):
                y, x = y_ - 2, x_ - 2
                vec = Vector2(vectors[y, x, 0], vectors[y, x, 1])
                if (vec.x == 0 and vec.y == 0):
                    draw[y, x] = 1
                    continue
                vec = vec / abs(vec)
                pos = Vector2(x + 0.5, y + 0.5)
                while (pos.x < x + 3 and pos.x > x - 2 and pos.y < y + 3 and pos.y > y - 2):
                    draw[pos.y, pos.x] = 1
                    pos += vec

        return draw


