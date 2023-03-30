#from scene import *
#import scene
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

#from trader import Trader

food_vals = [
    0.8,  # shallow
    0.1,  # deep
    0.4,  # alpine
    1.0,  # jungle
    0.8,  # forest
    0.8,  # grassland
    0.6,  # taiga
    -0.1,  # tundra
    -0.1,  # desert
]
food_vals = np.array(food_vals)

ZW = 12
RW = 10
UW = 5

grow_tiles = []
for x in range(ZW * 2 + 1):
    for y in range(ZW * 2 + 1):
        grow_tiles.append((x, y))


def get_order(radius=ZW):
    g_ord = grow_tiles[:]
    r.shuffle(g_ord)
    g_ord = [p for p in g_ord if abs(Vector2(p[0], p[1]) - Vector2(ZW, ZW)) < min(ZW, radius)]
    g_ord.sort(key=lambda p: abs(Vector2(p[0], p[1]) - Vector2(ZW, ZW)))
    return g_ord


urban_cap = [0, 5, 25]
rural_cap = [15, 20, 25]
level_zone = [5, 15, 50]


class City():
    def __init__(self, x, y, c):
        self.id = c.map.current_id
        c.map.current_id += 1

        self.zone = np.zeros((1 + ZW * 2, 1 + ZW * 2))
        self.Rzone = np.zeros((1 + ZW * 2, 1 + ZW * 2))
        self.Uzone = np.zeros((1 + ZW * 2, 1 + ZW * 2))

        self.food = 10

        self.map = c.map
        self.con = c
        if (c.capital == None):
            c.capital = self
        self.pos = Vector2(x, y)
        self.x = x
        self.y = y
        self.zone_capped = False
        self.level_capped = False
        self.dev_capped = False

        self.local_harvest = 0
        self.out_harvest = 0
        self.trans_harvest = 0
        self.net_harvest = 0

        self.local_income = 0
        self.out_income = 0
        self.trans_income = 0
        self.net_income = 0

        # passive
        # trade+trans+local-costs=net
        # reset trans

        # active
        # send trans
        # use net for decisions
        # decide things
        # reset net

        for x_ in [-1, 0, 1]:
            for y_ in [-1, 0, 1]:
                self.claim_tile(ZW + x_, ZW + y_)

        self.level = 0
        self.grow_order = get_order(level_zone[self.level])

        self.b_tiles = np.zeros(9)

        self.rural = 0
        self.claim_rural()
        self.urban = 0
        # village
        # town
        # city

        self.occupied = 0
        self.gar_size = 0

        self.alive = True
        # self.trader = None

        self.trade_city = None

    def passive(self):
        # harvest
        self.local_harvest = np.sum(self.b_tiles * food_vals)

        self.local_income = self.urban

        # trade
        if (self.trade_city != None and self.trade_city.alive):
            trade_income = int(self.trade_city.local_income / 5) + 1
            trade_harvest = int(self.trade_city.local_harvest / 5) + 1
        else:
            trade_income = 0
            trade_harvest = 0

        # get net
        self.total_harvest = self.local_harvest + trade_harvest + self.trans_harvest

        self.total_income = self.local_income + trade_income + self.trans_income

        # subtract costs
        self.net_harvest = self.total_harvest - (self.urban * 2)

        self.net_income = self.total_income - self.gar_size

        self.trans_harvest = 0
        self.trans_income = 0

    def active(self):
        # Transfer
        self.out_income = 0
        self.out_harvest = 0
        if (self.level < 2):
            _, n = self.nearest_town(self.level + 1)
            if (n != None):
                self.transfer(n)

        # Update totals
        self.food += self.net_harvest - self.out_harvest
        self.con.wealth += self.net_income - self.out_income

        # Grow
        self.grow()

        # Trade
        if (self.level > 0):
            self.trade()

        # Control Garrison
        if (self.level > 0):
            self.garrison()

    def garrison(self):
        des_gar = int(self.total_income * self.con.gar_rate)
        if (des_gar < self.gar_size):
            self.gar_size -= 1
        elif (des_gar > self.gar_size):
            self.gar_size += 1

    def transfer(self, nearest):
        h_rate = 1 if self.dev_capped else self.con.tran_rate
        if (self.net_harvest > 0):
            self.out_harvest = int(self.net_harvest * h_rate)
            nearest.trans_harvest += self.out_harvest
        if (self.net_income > 0):
            self.out_income = int(self.net_income * h_rate)
            nearest.trans_income += self.out_income

    def trade(self):
        if (self.trade_city == None or self.trade_timer <= 0 or not self.trade_city.alive):
            targets = []
            for c in self.map.all_civs:
                if (c.id in self.con.at_war):
                    continue
                if (c.id == self.con.id):
                    continue
                targets.extend(c.towns)
            s_t = sorted(targets, key=lambda x: abs(x.pos - self.pos))
            if (len(s_t) > 0):
                if (abs(s_t[0].pos - self.pos) < 50):
                    # t = Trader(self, s_t[0])
                    self.trade_city = s_t[0]
                    self.trade_timer = int(abs(s_t[0].pos - self.pos))
        else:
            self.trade_timer -= 1

    # else:
    #	res = self.trader.move()
    #	if(res != 0):
    #		self.trader = None

    def claim_tile(self, x, y):
        real_x = x + self.x - ZW
        real_y = y + self.y - ZW

        if (self.map.claims[real_y, real_x] != -1):
            return False

        assert (self.con.zone[real_y, real_x] == 0)
        assert (self.map.claims[real_y, real_x] == -1)

        self.zone[y, x] = 1
        self.con.zone[real_y, real_x] = 1
        self.map.claims[real_y, real_x] = self.con.id

        return True

    def nearest_town(self, level):
        min_dist = 100000
        min_city = None
        for c in self.con.settlements:
            if (c.level >= level):
                min_dist = min(min_dist, abs(c.pos - self.pos))
                min_city = c
        return min_dist, min_city

    def get_b(self, x, y):
        real_x = x + self.x - ZW
        real_y = y + self.y - ZW

        return np.argmax(self.map.biome_map[real_y, real_x])

    def claim_rural(self):
        self.rural += 1
        for x, y in self.grow_order:
            if (self.zone[y, x] == 1 and self.Rzone[y, x] == 0 and self.Uzone[y, x] == 0):
                self.Rzone[y, x] = 1
                self.b_tiles[self.get_b(x, y)] += 1
                return
        self.rural -= 1
        self.dev_capped = True

    # assert(False)

    def unclaim_rural(self):
        self.rural -= 1
        for x, y in reversed(self.grow_order):
            if (self.zone[y, x] == 1 and self.Rzone[y, x] == 1 and self.Uzone[y, x] == 0):
                self.Rzone[y, x] = 0
                self.b_tiles[self.get_b(x, y)] -= 1
                return
        # self.rural -= 1
        # self.dev_capped = True
        assert (False)

    def claim_urban(self):
        self.urban += 1
        for x, y in self.grow_order:
            if (self.zone[y, x] == 1 and self.Rzone[y, x] == 1 and self.Uzone[y, x] == 0):
                real_x = x + self.x - ZW
                real_y = y + self.y - ZW
                if (self.map.water[real_y, real_x] == 1):
                    continue
                self.Uzone[y, x] = 1
                self.Rzone[y, x] = 0
                self.b_tiles[self.get_b(x, y)] -= 1
                self.rural -= 1
                self.claim_rural()
                if (self.con.capital.urban < self.urban and self.occupied == 0):
                    self.con.capital = self
                return
        self.urban -= 1
        self.dev_capped = True
        return
        assert (False)

    def unclaim_urban(self):
        self.urban -= 1
        for x, y in reversed(self.grow_order):
            if (self.zone[y, x] == 1 and self.Rzone[y, x] == 0 and self.Uzone[y, x] == 1):
                real_x = x + self.x - ZW
                real_y = y + self.y - ZW
                if (self.map.water[real_y, real_x] == 1):
                    continue
                self.Uzone[y, x] = 0
                self.Rzone[y, x] = 1
                self.b_tiles[self.get_b(x, y)] += 1
                self.rural += 1
                self.unclaim_rural()
                return
        assert (False)

    def grow(self):
        food_cost = ((self.rural + self.urban) ** 2)
        if (self.food < 0):
            self.food += food_cost
            if (self.urban > 0):
                self.unclaim_urban()
            elif (self.rural > 0):
                self.unclaim_rural()
            else:
                self.log(f'city:{self.id} starved to death')
                self.map.raze(self)
            return

        if (self.food < food_cost * 2):
            return

        if (self.dev_capped):
            return

        urban_fail = False
        if (self.net_harvest >= 10):
            if (self.urban < urban_cap[self.level]):
                # self.urban += 1
                self.claim_urban()
            elif (self.level_capped or self.level == 2):
                urban_fail = True
            else:
                nearest, _ = self.nearest_town(self.level + 1)
                if (nearest < level_zone[self.level + 1]):
                    self.level_capped = True
                else:
                    self.level += 1
                    self.claim_urban()
                    # self.urban += 1
                    self.con.log(f'city:{self.id} grew to level:{self.level}')
                    if (self.level == 1):
                        self.con.towns.append(self)
                    if (self.level == 2):
                        self.con.cities.append(self)
                    self.grow_order = get_order(level_zone[self.level])
        rural_fail = False
        if (self.net_harvest < 10 or urban_fail):
            if (self.rural < rural_cap[self.level]):
                # self.rural += 1
                self.claim_rural()
            else:
                rural_fail = True
        if (rural_fail and urban_fail):
            self.dev_capped = True
            return

        self.food -= food_cost

        if (self.zone_capped):
            return

        grow = 3
        for x, y in self.grow_order:
            res = self.claim_tile(x, y)

            if (res == True):
                grow -= 1
            if (grow == 0):
                break
        if (grow > 0):
            self.zone_capped == True