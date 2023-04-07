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

from city import City, food_vals
from army import Army

ZW = 12
RW = 10
UW = 5

class Civ():
	def __init__(self, m, id):
		self.id = id
		self.wealth = 0
		self.income = 0

		self.capital = None
		self.settlements = []
		self.towns = []
		self.cities = []
		self.zone = np.zeros(m.slope.shape)
		self.map = m
		self.normcolor = np.random.randint(75, 200, 3)
		self.color = self.normcolor
		self.opinions = {}
		self.army = None
		self.army_size = 0
		self.marching = False
		self.at_war = []
		self.wealth_rate = 0
		self.level = 0
			#Tribe
			#Settlement
			#City-State
			#Empire
		self.alive = True

		self.atk_rate = 0.2
		self.gar_rate = 0.2
		self.tran_rate = 0.3

		self.culture = -1
		self.main_faith = -1

	def opin(self,src,val,rsn=0):
		self.opinions[src] = self.opinions.get(src, 0)+val

	def find_new_neighbors(self):
		for c in self.cities:
			for c_ in self.map.all_civs:
				if(c_.id in self.opinions):
					continue
				for c2 in c_.cities:
					if(abs(c.pos - c2.pos) > 50):
						continue
					if(self.id == c2.con.id):
						continue

					c.con.opin(c_.id, 0)
					self.log(f'civ:{self.id} met civ:{c_.id}')

	def take_turn(self):
		if(self.level == 2 and r.random() < 0.1):
			self.find_new_neighbors()


		for c in self.settlements:
			c.passive()

		self.income = 0
		for c in self.settlements:
			c.active()

		if(self.army == None):
			self.a_cost=self.army_size
		else:
			self.a_cost=2* self.army_size
		self.net_income = self. income - self.a_cost
		self.wealth += self.net_income

		self.recruit()

		#self.trade()
		self.action()

		s = 0
		for c in self.settlements:
			s += np.sum(c.zone)
		assert(s == np.sum(self.zone))

	def recruit(self):
		a_rate = (1 - self.gar_rate) * self.atk_rate
		while(self.a_cost < self.income * a_rate and self.wealth > 10):
			self.army_size += 1
			self.wealth -= 10

	def trade(self):
		for t in self.towns:
			t.trade()

	def grow(self):
		old = np.sum(self.zone)
		for c in self.settlements:
			c.grow()
			if(c.level > self.level):
				self.log(f'civ:{self.id} grew to level:{c.level}')
				self.level = c.level

	def log(self, m):
		f = open('./log.txt', 'a+')
		f.write(m + '\n')
		f.close()

	def action(self):
		if(self.wealth >= 100 and len(self.at_war) == 0):
			result = True#self.create_colony()
			if(result):
				self.log(f'civ:{self.id} created city:{self. settlements[-1].id}')
				self.wealth -= 100
		#create colony

		#declare war
		if(self.level >= 2 and len(self.at_war) == 0):
			for c_id in self.opinions:
				if(self.opinions[c_id] < -100):
					self.log(f'civ:{self.id} declares war against civ:{c_id}')
					self.at_war. append(c_id)
					self.opin(c_id,100)
		self.do_war()

	def do_war(self):
		if(self.wealth < 0):
			self.at_war = []
		if(len(self.at_war) == 0):
			self.army = None
			self.marching = False
		if(self.marching):
			res = self.army.move()
			if(res == 1):
				res2 = self.army.attack()
				if(res2 and len(self.at_war) > 0):
					self.army.tgt = self.find_target()
					self.army.move()
				elif(res2):
					self.map.all_armies. remove(self.army)
					self.marching = False
			elif(res == 2):
				self.army.tgt = self.find_target()

		if(len(self.cities) == 0):
			self.at_war = []

		#attack city
		if(len(self.at_war) > 0 and self.marching == False and len(self.cities) > 0 and self.army_size > 0):
			source = r.choice(self.cities)
			target_civ = self.map.id2civ[r.choice(self.at_war)]
			self.target = self.find_target()
			self.army = Army(source, self, self.target)
			self.army.move()
			self.marching = True

	def find_target(self):
		enemies = []
		for e in self.at_war:
			enemies.extend (self.map.id2civ[e].settlements)
		if(self.army != None):
			pos = self.army.pos
		else:
			pos = r.choice (self.cities).pos
		s_e = sorted(enemies, key=lambda x: abs(x.pos - pos))
		if(len(s_e) == 0):
			self.at_war = []
		return s_e[0]

	def get_loc_score(self, x, y):
		return np.sum(self.map.biome_map[x-3:x+4,x-3:x+4] * food_vals[None,None,:])

	def create_colony(self):
		best_score = -1
		best_loc = -1
		for i in range(100):
			angle = r.uniform(0,math.pi*2)
			dist = r.uniform(5, 8)
			start = r.choice(self.settlements)
			vec = Vector2(math.cos(angle), math.sin(angle))
			new_pos = (vec * dist) + start.pos
			x = int(new_pos.x)
			y = int(new_pos.y)
			if(self.map.valid_city(x,y,self.id)):
				score = self.get_loc_score(x, y)
				if(score > best_score and score > 0):
					best_score = score
					best_loc = (x,y)

		if(best_score != -1):
			x,y = best_loc
			new_city = City(x, y, self)
			self.map.all_cities. append(new_city)
			self.settlements.append (new_city)
			self.map.id2city[new_city.id] = new_city
			return True
		return False