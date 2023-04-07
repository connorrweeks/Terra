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

from terrain_maker import TerrainMaker

from city import City
from civ import Civ

ZW = 12
RW = 10
UW = 5

faith_names = [
	'sun', #- more unity
	'sea', #- more sea trade
	'sky', #- faster conversion
	'earth', #- faster growth
	'death', #- avoid disasters
	'life', #- more food
]

culture_names = [
	'partiers', #- more happiness
	'bureaucrats', #- more development in capital
	'raiders', #- gain more wealth from conquest
	'traders', #- gain more wealth from trade
	'miners', #- bonus from natural resources
	'priests', #- more faith
	'thinkers', #- more
	'warriors', #
]

class CivMap(TerrainMaker):
	def __init__(self, w, h):
		w = int(w)
		h = int(h)
		super().__init__(w, h)
		self.current_id = 0
		self.all_cities = []
		self.all_civs = []
		self.all_armies = []
		self.all_traders = []
		self.id2civ = {}
		self.id2city = {}
		self.claims = np.full((h,w),-1)

	def make_a_city(self):
		tries = 100
		for i in range(tries):
			x=r.randrange(self.width)
			y=r.randrange(self.height)
			pos = Vector2(x,y)

			if(self.water[y,x] == 1):
				continue

			nearby = False
			for c in self.all_cities:
				if(abs(pos-c.pos) <= 5):
					nearby = True
					break
			if(nearby): continue

			controlled = False
			for c in self.all_civs:
				if(c.zone[y,x] == 1):
					controlled = True
					break
			if(controlled): continue

			new_civ = Civ(self, self.current_id)
			self.current_id += 1
			new_city=City(x,y,new_civ)
			new_civ.settlements.append(new_city)
			self.all_cities.append (new_city)
			self.all_civs.append (new_civ)
			self.id2city[new_city.id] = new_city
			self.id2civ[new_civ.id] = new_civ

			break

	def full_turn(self):
		#self.find_neighbors()
		for c in self.all_civs:
			c.take_turn()

		if(r.random() < 1 - len(self.all_civs) / 200):
			self.make_a_city()

	def raze(self, tgt):
		dfd = tgt.con
		if(tgt in dfd.cities):
			dfd.cities.remove(tgt)
		if(tgt in dfd.towns):
			dfd.towns.remove(tgt)
		assert(tgt in dfd.settlements)
		dfd.settlements.remove(tgt)
		dfd.zone[tgt.y-ZW:tgt.y+ZW+1, tgt.x-ZW:tgt.x+ZW+1] -= tgt.zone
		self.claims[tgt.y-ZW:tgt.y+ZW+1, tgt.x-ZW:tgt.x+ZW+1] -= (tgt.zone * (dfd.id + 1))
		if(len(dfd.settlements) == 0):
			atk.log(f'civ:{dfd.id} was destroyed by civ:{atk.id}')
			dfd.alive = False
			self.delete_civ (dfd.id)
		else:
			if(dfd.capital == tgt):
				dfd.capital = sorted(dfd.settlements, key=lambda x:-x.urban)[0]

	def cap(self, atk, dfd, tgt):
		if(tgt in dfd.cities):
			dfd.cities.remove(tgt)
		if(tgt in dfd.towns):
			dfd.towns.remove(tgt)
		assert(tgt in dfd.settlements)
		dfd.settlements.remove(tgt)
		dfd.zone[tgt.y-ZW:tgt.y+ZW+1, tgt.x-ZW:tgt.x+ZW+1] -= tgt.zone
		self.claims[tgt.y-ZW:tgt.y+ZW+1, tgt.x-ZW:tgt.x+ZW+1] -= (tgt.zone * (dfd.id + 1))
		if(tgt.level == 0):
			atk.log(f'civ:{atk.id} destroyed city:{tgt.id} from civ:{dfd.id}')
			tgt.alive = False
			self.all_cities.remove(tgt)
		else:
			atk.log(f'civ:{atk.id} captured city:{tgt.id} from civ:{dfd.id}')
			tgt.occupied = 1
			if(tgt.level == 2):
				atk.cities.append(tgt)
			if(tgt.level == 1):
				atk.towns.append(tgt)
			atk.settlements.append(tgt)

			tgt.con = atk

			atk.zone[tgt.y-ZW:tgt.y+ZW+1, tgt.x-ZW:tgt.x+ZW+1]+= tgt.zone

			self.claims[tgt.y-ZW:tgt.y+ZW+1, tgt.x-ZW:tgt.x+ZW+1] += (tgt.zone * (atk.id + 1))

		if(len(dfd.settlements) == 0):
			atk.log(f'civ:{dfd.id} was destroyed by civ:{atk.id}')
			dfd.alive = False
			self.delete_civ (dfd.id)
		else:
			if(dfd.capital == tgt):
				dfd.capital = sorted(dfd.settlements, key=lambda x:-x.urban)[0]

	def valid_city(self, x, y, id):
		if(x < 2 or y < 2):
			return False
		if(x >= self.width-2):
			return False
		if(y >= self.height-2):
			return False
		if(self.water[y,x] == 1):
			return False
		if(np.sum(self.claims[y-1:y+2, x-1:x+2] == -1) + np.sum(self.claims[y-1:y+2, x-1:x+2] == id) != 9):
			return False
		pos = Vector2(x,y)
		for c in self.all_cities:
			if(abs(c.pos - pos) < 5):
				return False
		for c in self.all_civs:
			if(c.zone[y,x] == 1):
				return False
		return True

	def delete_civ(self, id):
		self.all_civs.remove (self.id2civ[id])

		self.claims -= self.id2civ[id].zone * (id + 1)

		#self.id2civ.pop(id)

		for c in self.all_civs:
			if(id in c.opinions):
				c.opinions.pop(id)
			if(id in c.at_war):
				c.at_war.remove(id)