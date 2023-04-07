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

class Army():
	def __init__(self, base, con, tgt):
		self.base = base
		self.pos = base.pos
		self.con = con
		self.tgt = tgt
		self.speed = 1.0
		self.con.map.all_armies. append(self)
		self.t = 0

	def move(self):
		if(self.tgt.alive == False):
			return 2
		self.t += 1
		self.vec = self.tgt.pos - self.pos
		self.vec /= abs(self.vec)
		self.pos += self.vec * self.speed
		self.y = int(self.pos.y)
		self.x = int(self.pos.x)
		if(abs(self.pos - self.tgt.pos)<1):
			return 1
		return 0

	def attack(self):
		self.tgt.con.opin (self.con.id, -30)
		if(self.con.army_size < self.tgt.urban):
			self.con.army_size = 0
			self.con.map.all_armies. remove(self)
			self.con.army = None
			self.con.marching = False
			self.con.at_war = []
			self.con.log(f'civ:{self.con.id} lost army at city:{self.tgt.id} ending the war')
			return False
		else:
			self.con.army_size -= self.tgt.urban
			self.con.map.cap(self.con, self.tgt.con, self.tgt)
			self.con.log(f'army t:{self.t} size:{self.con.army_size}')
			return True