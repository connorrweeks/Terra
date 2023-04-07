#from scene import *
#import scene
#import sound
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
from vect2 import Vector2
from sklearn.cluster import KMeans

#A = Action

CON_MULT = 1.0
REG_MULT = 0.6
LOC_MULT = 0.4
TER_MULT = 0.2

BIOME_DIFFUSE = 45
RAIN_DIFFUSE = 200

AIR_FIX_AND_DIFFUSE = 600
AIR_JUST_FIX = 50

FIX_AND_DIFFUSE = 600
JUST_FIX = 50

RIDGE_WIDTH = 25

MICRO_CUTOFF = 100
PROVINCE_SIZE = 50

BIOME_AREA_SEPARATOR = 15

step_map = np.vectorize(lambda x, k: math.ceil(x * k) / k)
bin_map = np.vectorize(lambda x: 1.0 if x > 0.0 else 0.0)

def seed(set_seed=None):
	global world_seed
	if(set_seed == None):
		world_seed = r.randrange(0,1000)
	else:
		world_seed = set_seed
	print("world_seed", world_seed)
	r.seed(world_seed)
	np.random.seed(world_seed)
	return world_seed

def diffuse(arr, decay=1.0, diff=0.2):
		#decay = 0.95
		#diff = 0.25
		if(len(arr.shape) == 2):
			s = (arr.shape[0]+2, arr.shape[1]+2)
		else:
			s = (arr.shape[0]+2, arr.shape[1]+2,arr.shape[2])
		new_arr = np.zeros(s)

		for x in [-1, 0, 1]:
			for y in [-1, 0, 1]:
				if(x == 0 and y == 0):
					mult = 1 - diff
				else:
					mult = diff / 8
				new_arr[1+y:arr.shape[0]+1+y, 1+x:arr.shape[1]+1+x] += arr * mult
		return new_arr[1:-1,1:-1] * decay

def vec2ang(vec):
	unit_vec = vec / np.linalg.norm(vec)
	dot_product = np.dot(unit_vec, [1,0])
	angle = np.arccos(dot_product)
	return angle

class TerrainMaker():
	def __init__(self, width=-1, height=-1):
		self.width = int(width)
		self.height = int(height)

		self.t0 = time.perf_counter()

		self.np_attr = [
		'elevation',
		'orig_elevation',
		'slope',
		'slope_vectors',
		'temperature',
		'water',
		'sunlight',
		'cloud_map',
		'currents',
		'air_currents',
		'biome_map',
		'rainfall',
		'rivers',
		'run_off',
		'island',
		'bio_ids',
		'provinces',
		'ocean',
		'areas',
		]

		self.attr = [
			'height',
			'width',
			'equator',
			'deposits',
			'island_sizes',
			'bio_sizes',
			'bio_types',
			'island_provs',
			'province_locs',
			'area_locs',
			'num_provs',
			'num_areas',
			'prov_areas',
			'prov_types',
			'area_provs',
		]

		

	def load_pickle(self, n):
		p = f'./saves/{n}'
		for a in self.np_attr:
			if(os.path.exists(f'{p}/{a}.npy') == False): continue
			obj=np.load(f'{p}/{a}.npy')
			setattr(self, a, obj)

		for a in self.attr:
			if(os.path.exists(p+'/'+a) == False): continue
			with open(p+'/'+a, 'rb') as f:
				obj = pickle.load(f)
			setattr(self, a, obj)
		
		self.color_noise = np.random.uniform(-1, 1, (self.height, self.width,3))

	def save_pickle(self, n):
		p = f'./saves/{n}'
		if(os.path.isdir(p)):
			shutil.rmtree(p)
		os.makedirs(p)

		for a in self.np_attr:
			try:
				obj = getattr(self, a)
				np.save(p+'/'+a, obj)
			except:
				print("Save failed:", a)

		for a in self.attr:
			try:
				obj = getattr(self, a)
				with open(p+'/'+a, 'wb+') as f:
					pickle.dump(obj, f)
			except:
				print("Save failed:", a)
			

	def generate(self, seed=0, skip=0):
		#self.seed = r.randrange(0, 100) * 100
		self.seed = world_seed
		self.t0 = time.perf_counter()

		if(skip < 1): self.phase1()
		if(skip < 2): self.phase2()
		if(skip < 3): self.phase3()
		if(skip < 4): self.phase4()
		if(skip < 5): self.phase5()
		if(skip < 6): self.phase6()

	def phase1(self):
		self.elevation = np.zeros((self.height, self.width))
		self.biome_map = np.zeros((self.height, self.width, 9))
		self.color_noise = np.random.uniform(-1, 1, (self.height, self.width,3))

		self.elevation = self.noise_map()
		print(f"Generating Noise Map - {time.perf_counter()-self.t0:.2f}")

		self.elevation = self.crop(self.elevation)
		print(f"Cropping Map - {time.perf_counter()-self.t0:.2f}")

		self.add_ridges()
		print(f"Adding Ridges - {time.perf_counter()-self.t0:.2f}")

		self.slope_vectors = self.make_slope_vectors(self.elevation)
		self.slope = np.linalg.norm(self.slope_vectors, axis=2)
		print(f"Calculating Slope Vectors - {time.perf_counter()-self.t0:.2f}")

		self.orig_elevation = self.elevation

	def phase2(self):
		self.build_currents()
		print(f"Stirring Ocean - {time.perf_counter()-self.t0:.2f}")

		self.build_atmosphere()
		print(f"Blowing Air - {time.perf_counter() - self.t0:.2f}")

	def phase3(self):
		self.set_sunlight()
		print(f"Setting Equator - {time.perf_counter()-self.t0:.2f}")
		self.simulate_clouds()
		print(f"Simulating Clouds - {time.perf_counter()-self.t0:.2f}")

	def phase4(self):
		self.elevation = self.orig_elevation[:,:]
		self.simulate_erosion()
		print(f"Simulating Erosion - {time.perf_counter()-self.t0:.2f}")

		self.define_oceans()
		print(f"Defining Ocean - {time.perf_counter()-self.t0:.2f}")

		self.rivers = np.zeros(self.slope.shape)
		self.build_rivers()
		print(f"Building Rivers - {time.perf_counter()-self.t0:.2f}")


	def phase5(self):
		self.create_biomes()
		print(f"Outlining Biomes - {time.perf_counter()-self.t0:.2f}")

		self.eliminate_microbiomes()
		print(f"Eliminating Micro-Biomes - {time.perf_counter()-self.t0:.2f}")

		self.scatter_minerals()
		print(f"Scattering Minerals - {time.perf_counter()-self.t0:.2f}")

	def phase6(self):
		self.find_islands()
		print(f"Finding Islands - {time.perf_counter()-self.t0:.2f}")

		self.find_bodies()
		print(f"Finding Bodies - {time.perf_counter()-self.t0:.2f}")

		self.segment_provinces()
		print(f"Segmenting provinces - {time.perf_counter()-self.t0:.2f}")

		self.group_areas()
		print(f"Grouping areas - {time.perf_counter()-self.t0:.2f}")



	def group_areas(self):
		self.num_areas = int(self.num_provs / 10)
		self.area_provs = {i:[] for i in range(self.num_areas)}
		self.prov_areas = []
		km = KMeans(n_clusters=self.num_areas, random_state=0)
		one_hot_biomes = np.zeros((len(self.prov_types), max(self.prov_types)+1), dtype=int)
		one_hot_biomes[np.arange(len(self.prov_types)),np.array(self.prov_types)] = BIOME_AREA_SEPARATOR
		points = np.concatenate([np.array(self.province_locs), one_hot_biomes], axis=1)

		preds = km.fit_predict(points)
		for i in range(self.num_provs):
			self.area_provs[preds[i]].append(i)
			self.prov_areas.append(preds[i])
		self.area_locs = [(km.cluster_centers_[i, 0], km.cluster_centers_[i, 1]) for i in range(self.num_areas)]

		prov_no_water = self.provinces + self.water
		self.areas = np.array(self.prov_areas)[prov_no_water.astype(int)]
		self.areas = ((self.areas * (1 - self.water)) - self.water).astype(int)
		print("Number of areas -", self.num_areas)

	def segment_provinces(self):
		self.provinces = np.full(self.water.shape,-1)
		self.province_locs = []
		self.prov_types = []
		self.island_provs = {isle:set() for isle in range(len(self.island_sizes))}
		points = {i:[] for i in range(len(self.bio_sizes))}
		for y in range(self.height):
			for x in range(self.width):
				if(self.bio_ids[y, x] == -1): continue
				points[self.bio_ids[y, x]].append([x, y])
		points = [np.array(points[i]) for i in range(len(points))]
		tot_provs = 0
		for i, size in enumerate(self.bio_sizes):
			k = max(1, int(size / PROVINCE_SIZE))
			assert(len(points[i]) > 0)
			km = KMeans(n_clusters=k, random_state=0)
			preds = km.fit_predict(points[i])
			centers = km.cluster_centers_
			self.prov_types.extend([self.bio_types[i]] * k)
			for j, (x, y) in enumerate(points[i]):
				self.provinces[y, x] = preds[j] + tot_provs
			self.province_locs.extend([(int(centers[j,0]),int(centers[j,1])) for j in range(k)])
			tot_provs += k
		for y in range(self.height):
			for x in range(self.width):
				isle = self.island[y, x]
				if(isle == -1): continue
				self.island_provs[isle].add(self.provinces[y, x])
			
		self.num_provs = tot_provs
		print("Number of provinces -", self.num_provs)

	def eliminate_microbiomes(self):
		self.find_bios()
		
		b_neighbors = {x:set() for x in range(len(self.bio_sizes))}
		for y in range(self.height-1):
			for x in range(self.width-1):
				if(self.bio_ids[y, x] == -1): continue
				#if(self.bio_ids[y, x+1] == -1): continue
				#if(self.bio_ids[y+1, x] == -1): continue
				if(self.bio_ids[y, x] != self.bio_ids[y+1, x] and self.bio_ids[y+1, x] != -1):
					b_neighbors[self.bio_ids[y, x]].add(self.bio_ids[y+1, x])
					b_neighbors[self.bio_ids[y+1, x]].add(self.bio_ids[y, x])
				if(self.bio_ids[y, x] != self.bio_ids[y, x+1] and self.bio_ids[y, x+1] != -1):
					b_neighbors[self.bio_ids[y, x]].add(self.bio_ids[y, x+1])
					b_neighbors[self.bio_ids[y, x+1]].add(self.bio_ids[y, x])

		elims = 0
		for i, size in enumerate(self.bio_sizes):
			if(size >= MICRO_CUTOFF): continue
			if(len(b_neighbors[i]) == 0): continue
			sorted_neighors = sorted([(j, self.bio_sizes[j]) for j in b_neighbors[i]], key=lambda x: -x[1])
			j, _ = sorted_neighors[0]
			#print(i, j)
			if(self.bio_sizes[j] <= self.bio_sizes[i]): continue
			self.bio_types[i] = self.bio_types[j]
			elims += 1

		b_prime = np.argmax(self.biome_map, axis=2)
		b_max = np.max(self.biome_map, axis=2)
		for y in range(self.height):
			for x in range(self.width):
				if(self.bio_ids[y,x] == -1): continue
				new_b = self.bio_types[self.bio_ids[y, x]]
				if(b_prime[y, x] == new_b): continue
				self.biome_map[y, x, :] = [0] * 9
				self.biome_map[y, x, new_b] = 1.0
		self.find_bios()
		print(f"Biomes eliminated - {elims}")

	def find_bios(self):
		b_prime = np.argmax(self.biome_map, axis=2)
		self.bio_sizes = []
		self.bio_types = []
		self.bio_ids = np.full(self.water.shape,-1)

		for y in range(self.height):
			for x in range(self.width):
				if(self.bio_ids[y,x] == -1 and self.water[y,x] == 0):
					size = self.explore(x, y, len(self.bio_sizes), b_prime, self.bio_ids)
					self.bio_sizes.append(size)
					self.bio_types.append(b_prime[y, x])

	def find_bodies(self):
		self.body_sizes = []
		self.body = np.full(self.water.shape,-1)

		for y in range(self.height):
			for x in range(self.width):
				if(self.body[y,x]==-1 and self.water[y,x]==1):
					size=self.explore(x, y, len(self.body_sizes), self.water, self.body)
					self.body_sizes.append(size)

	def find_islands(self):
		self.island_sizes = []
		self.island = np.full(self.water.shape,-1)

		for y in range(self.height):
			for x in range(self.width):
				if(self.island[y,x]==-1 and self.water[y,x]==0):
					size = self.explore(x,y, len(self.island_sizes), self.water, self.island)
					self.island_sizes.append(size)

	def explore(self, x, y, id, in_arr, out_arr):
		corr_val = in_arr[y, x]
		frontier = [(x,y)]
		end = 1
		size = 0
		while(end > 0):
			x,y = frontier[end - 1]
			end -= 1
			for dx in [-1,0,1]:
				for dy in [-1,0,1]:
					if(dx+x<0): continue
					if(dy+y<0): continue
					if(dx+x >= self.width): continue
					if(dy+y >= self.height): continue
					if(out_arr[dy+y, dx+x] != -1): continue
					if(in_arr[dy+y, dx+x] != corr_val): continue
					if(end == len(frontier)):
						frontier.append ((dx+x,dy+y))
					else:
						frontier[end] = (dx+x,dy+y)
					out_arr[dy+y,dx+x]=id
					end += 1
					size += 1
		return size

	def define_oceans(self):
		self.ocean = np.full(self.slope.shape, 0)
		self.explore(0, 0, 1, self.water, self.ocean)

	def build_rivers(self):
		inertia = 0.01
		self.rivers = np.zeros(self.slope.shape)
		self.run_off = np.zeros(self.elevation.shape)
		num_rivers = 0
		paths = []
		for drop in range(4000):
			x =r.randrange(self.width)
			y =r.randrange(self.height)
			if(r.random() < self.rainfall[y,x] or self.water[y,x] == 1):
				continue
			path = []
			mem = set()
			pos= Vector2(x,y)
			dir = Vector2(0,0)
			to_water = False
			for step in range(self.height):
				#if((pos.x, pos.y) in mem):
					#break
				mem.add((pos.x,pos.y))
				path.append((int(pos.x), int(pos.y)))
				slopex, slopey = self.get_slope(pos.x, pos.y)
				dir_x = (slopex * (1.0-inertia)) + (inertia * dir.x)
				dir_y = (slopey * (1.0-inertia)) + (inertia * dir.y)
				dir = Vector2(dir_x,dir_y)
				if(dir.x == 0 and dir.y == 0):
					break
				dir = dir / abs(dir)
				pos -= dir

				self.run_off[int(pos.y),int(pos.x)] = 1
				if(self.ocean[int(pos.y), int(pos.x)] == 1):
					to_water = True
					path.append((int(pos.x), int(pos.y)))
					break
				elif(self.water[int(pos.y), int(pos.x)] == 1):
					break
			if(not to_water):
				continue

			if(len(path) > 10):
				paths.append(path)

		num_rivers = r.randint(8,12)

		leng = [abs(Vector2(x[-1][0],x[-1][1])-Vector2(x[0][0],x[0][1])) for x in paths]
		scores = [(leng[i] * leng[i] / len(x), x) for i,x in enumerate(paths)]
		scores = sorted(scores, key=lambda x:-x[0])
		paths = [x[1] for x in scores]
		for path in paths[:num_rivers]:
			size = 0.5
			start = path[0]
			for p in path:
				minx = max(0,p[0]-int(size))
				maxx = min(self.width-1,p[0]+int(size+1.5))
				miny = max(0,p[1]-int(size))
				maxy = min(self.height-1,int(p[1]+size+1.5))
				for x in range(minx,maxx):
					for y in range(miny,maxy):
						self.rivers[y,x] = 1
						self.water[y,x] = 1

				size = (abs(Vector2(start[0], start[1])-Vector2(p[0], p[1])) / 75) + 0.5

		print(f'Adding {min(len(paths),num_rivers)}/{num_rivers} rivers')

	def simulate_erosion(self):
		self.diffs = []
		self.run_off = np.zeros(self.elevation.shape)
		total_drops = 4000
		for i_ in range(total_drops):
			x =r.randrange(self.width)
			y =r.randrange(self.height)
			if(r.random() < self.rainfall[y,x]):
				continue
			self.single_drop(x,y)
		for j_ in range(5):
			self.elevation = diffuse(self.elevation)
		self.slope = np.linalg.norm(self.slope_vectors, axis=2)
		#print(np.sum(self.run_off))

		self.elevation = self.elevation / np.max(self.elevation)

		self.water = 1 - bin_map(self.elevation.clip(0,1))

	def single_drop(self, x, y):
		pos = Vector2(x, y)
		accum = 0
		max_steps = 500
		size = 2
		sedimentCapacityFactor = 4.0
		minSedimentCapacity = 0.01
		gravity = 1
		erodeSpeed = 0.2
		depositSpeed = 0.5
		sediment = 0
		mem = set()
		speed = 1
		inertia = 0.01
		dir = Vector2(0,0)
		old_sum = np.sum(self.elevation)
		for step in range(max_steps):
			if(self.elevation[int(pos.y),int(pos.x)] < 0):
				break
			if((pos.x, pos.y) in mem):
				break
			mem.add((pos.x,pos.y))

			slope_x, slope_y = self.get_slope(pos.x, pos.y)
			dir_x = (slope_x * (1.0-inertia)) + (inertia * dir.x)
			dir_y = (slope_y * (1.0-inertia)) + (inertia * dir.y)
			dir = Vector2(dir_x,dir_y)
			if(dir.x == 0 and dir.y == 0):
				break
			dir = dir / abs(dir)

			n_pos = pos - (dir)
			deltaHeight = np.average(self.elevation[int(n_pos.y):int(n_pos.y+2),int(n_pos.x):int(n_pos.x+2)]) - np.average(self.elevation[int(pos.y):int(pos.y+2),int(pos.x):int(pos.x+2)])
			#if(deltaHeight > 0.05):
			#	break

			sedimentCapacity = max(-deltaHeight * speed, minSedimentCapacity)

			if(sediment >= sedimentCapacity or deltaHeight > 0):

				if(deltaHeight > 0):
					amountToDeposit = min(deltaHeight, sediment)
				else:
					amountToDeposit = min((sediment - sedimentCapacity) * depositSpeed, sediment)

				self.elevation [int(pos.y):int(pos.y+2),int(pos.x):int(pos.x+2)] += amountToDeposit / 4

				sediment -= amountToDeposit
			else:
				amountToErode = min ((sedimentCapacity - sediment) * erodeSpeed, -deltaHeight)

				self.elevation [int(pos.y-size):int(pos.y+size+2),int(pos.x-size):int(pos.x+size+2)] -= amountToErode / ((size+size+2) ** 2)

				sediment += amountToErode

			pos = Vector2(n_pos)
			if(pos.x < 0 or pos.y < 0 or pos.x >= self.width or pos.y >= self.height):
				break
			self.run_off[int(pos.y), int(pos.x)] += 1

			if((speed * speed) + (deltaHeight * gravity) < 0):
				break
			speed = math.sqrt((speed * speed) + (deltaHeight * gravity))

		self.elevation [int(pos.y):int(pos.y+2),int(pos.x):int(pos.x+2)] += sediment / 4

	def get_slope(self, x, y):
		slope_x = self.elevation [int(y),int(x+1)] - self.elevation [int(y),int(x)] + self.elevation [int(y+1),int(x+1)]- self.elevation [int(y+1),int(x)]

		slope_y = self.elevation [int(y+1),int(x)] - self.elevation [int(y),int(x)] + self.elevation [int(y+1),int(x+1)]-self.elevation [int(y),int(x+1)]

		return slope_x/2, slope_y/2

	def get_slope2(self, x, y):
		if(self.elevation[y,x+1] < self.elevation[y,x-1]):
			slope_x = self.elevation[y,x-1] - self.elevation[y,x]
		else:
			slope_x = self.elevation[y,x] - self.elevation[y,x+1]
		if(self.elevation[y+1,x] < self.elevation[y-1,x]):
			slope_y = self.elevation[y-1,x] - self.elevation[y,x]
		else:
			slope_y = self.elevation[y,x] - self.elevation[y+1,x]
		return slope_x, slope_y

	def scatter_minerals(self):
		to_add = []
		for m in range(5):
			to_add.extend([m] * r.randint(1,2))
		r.shuffle(to_add)

		self.deposits = []
		for min in to_add:
			for i_ in range(100):
				x = r.randrange(self.width)
				y = r.randrange(self.height)

				if(self.elevation[y,x] < r.random()): continue
				nearby = False
				for d in self.deposits:
					if(abs(Vector2(x,y) - Vector2(d[0],d[1])) < 50):
						nearby = True
						break
				if(nearby): continue
				self.deposits.append((x,y,min))
				break

	def create_biomes(self):
		biome_map = np.zeros((self.height,self.width,9))
		c = 0
		humidity = (self.temperature + self.rainfall) / 2
		dry_cold = (self.rainfall - self.temperature) / 2
		for y in range(self.height):
			for x in range(self.width):
				if(self.elevation[y,x] < -0.25):
					b=1#deep ocean
				#elif(self.elevation[y,x] < 0.0 or self.rivers[y,x] == 1.0):
				elif(self.elevation[y,x] < 0.0 or self.rivers[y,x] == 1):
					b=0#shallow
				elif(self.elevation[y,x] > 0.6):
					b=2#alpine
				elif(humidity[y,x]>0.75):
					b=3#rainforest
				elif(humidity[y,x]>0.62):
					b=4#forest
				elif(humidity[y,x]>0.48 and dry_cold[y,x]<-.25):
					b=5#grasslands
				elif(humidity[y,x]>0.48):
					b=6#taiga
				elif(dry_cold[y,x]<-.15):
					b=7#desert
				elif(dry_cold[y,x]<-.1):
					b=5#grasslands
				else:
					b=8#tundra
				biome_map[y,x,b] = 1.0

		b_sum = np.sum(biome_map, axis=2)
		#diffuse biomes
		for i_ in range(BIOME_DIFFUSE):
			for b in range(9):
				biome_map[:,:,b] = diffuse(biome_map[:,:,b])
		for b in range(2):
			biome_map[:,:,b] *= self.water
		for b in range(2,9):
			biome_map[:,:,b] *= 1 - self.water
		b_sum = np.sum(biome_map, axis=2)
		assert(np.min(b_sum) > 0)
		for b in range(9):
			biome_map[:,:,b] /= b_sum
		self.biome_map = biome_map
		print(np.average(self.rivers))

	def add_ridges(self):
		total_ridges = []
		num = r.randrange(2, 4)
		starters = []
		for y in range(self.height):
			for x in range(self.width):
				if(self.elevation[y,x] > 0.5):
					starters.append((x,y))
		for i_ in range(10):
			if(len(total_ridges) >= num):
				break
			start = r.choice(starters)

			nearby = False
			for p in total_ridges:
				if(abs(Vector2(p[0],p[1])-Vector2(p[0],p[1])) < 200):
					nearby = True
					break
			if(nearby):
				continue
			if(self.single_ridge_attempt(start)):
				total_ridges.append(start)
		for i in range(10):
			self.elevation = diffuse(self.elevation,diff=0.2,decay=1)
		self.elevation /= np.max(self.elevation)
		print('Total Ridges:',len(total_ridges))

	def single_ridge_attempt(self, start):
		x = start[0]
		y = start[1]
		success = False

		for k_ in range(40):
			#try angles
			s_angle = r.random() * 2 * math.pi
			ang_cha = ((r.random()*0.005) + 0.002) * r.choice([-1,1])

			length = 0
			#probe pass
			pos = Vector2(x,y)
			points = self.run_pass(pos, s_angle, ang_cha)

			points += self.run_pass(pos, (math.pi * 2)+s_angle, ang_cha)

			if(len(points) > 250):
				success = True
				break
		if(not success):
			return False

		points.append(pos)

		points = [(int(p.x),int(p.y)) for p in points]
		points = set(points)
		all = {}
		for p in points:
			p = Vector2(p[0], p[1])
			for x in range(-RIDGE_WIDTH,RIDGE_WIDTH+1):
				for y in range(-RIDGE_WIDTH,RIDGE_WIDTH+1):
					if(p.x+x<0 or p.x+x>=self.width or p.y+y<0 or p.y+y>=self.height):
						continue
					dist = abs(Vector2(x,y))
					if(dist > RIDGE_WIDTH):
						continue
					all[(p.x+x,p.y+y)] = max(all.get((p.x+x,p.y+y),0), (1.0 - (dist / RIDGE_WIDTH)))
		for p in all:
			#dist = abs()
			self.elevation[p[1], p[0]] = self.elevation[p[1], p[0]] + (all[p] * 0.5) + r.uniform(-0.05, 0.05)
		#print(len(points))

		return True

	def run_pass(self, pos, s_ang, c_ang):
		poss = []
		for step in range(1000):
			vec = Vector2(math.cos(s_ang), math.sin(s_ang))
			s_ang += c_ang
			pos += vec
			poss.append(pos)
			if(self.elevation[int(pos.y),int(pos.x)] < 0.1):
				return poss
		return poss

	def simulate_clouds(self):
		#num_clouds = 1000
		num_clouds = int(self.elevation.shape[0] * self.elevation.shape[1] / 320)
		num_steps = 300
		cloud_speed = 0.5
		real_clouds = 0
		cloud_size = 3.0

		self.cloud_map = np.zeros(self.slope.shape)
		self.rainfall = np.zeros(self.slope.shape)
		self.temperature = self.sunlight[:,:] / 2
		self.temperature -= self.elevation[:,:] / 20

		for i in range(num_clouds):
			start = Vector2(r.randrange(self.width), r.randrange(self.height))
			vel = self.currents[start.y, start.x]
			vel = Vector2(vel[1], vel[0])

			if(r.random() > self.sunlight[int(start.y), int(start.x)] ** 1):
				continue
			if(self.water[int(start.y), int(start.x)] == 0):
				continue
			real_clouds += 1

			self.cloud_map[start.y, start.x] = 1
			size = 4
			temp = self.sunlight[start.y, start.x]
			pos = Vector2(start.x, start.y)
			for step in range(num_steps):
				if(abs(vel) == 0):
					break
				pos += (cloud_speed * vel) / abs(vel)
				vel = vel + Vector2(self.air_currents[int(start.y), int(start.x)] / (step+1))
				vel.x += (r.random()-0.5)*0.1
				vel.y += (r.random()-0.5)*0.1
				if(pos.x<0 or pos.y<0 or pos.x >= self.width or pos.y >= self.height):
					break
				if(self.water[int(pos.y),int(pos.x)] == 1):
					size += self.sunlight[int(pos.y), int(pos.x)] * cloud_size
				else:
					size -= (self.elevation[int(pos.y), int(pos.x)] ** 3) * cloud_size

				if(size < 0):
					break

				miny = max(0, int(pos.y-math.sqrt(size)))
				maxy = min(self.height, int(pos.y+math.sqrt(size)))
				minx = max(0, int(pos.x-math.sqrt(size)))
				maxx = min(self.width, int(pos.x+math.sqrt(size)))

				self.rainfall[miny:maxy, minx:maxx] += max(0, 0.2 + self.elevation[int(pos.y),int(pos.x)])

				inertia = 0.7
				self.temperature[miny:maxy, minx:maxx] = (self.temperature[miny:maxy, minx:maxx] * inertia) + (temp * (1 - inertia))
				self.cloud_map[int(pos.y), int(pos.x)] = 1

		print('max rainfall', np.max(self.rainfall))
		print('max land rainfall', np.max((1 - self.water) * self.rainfall))
		print('real clouds', real_clouds)
		print('average temp', np.average(self.temperature))

		self.rainfall = np.sqrt(self.rainfall)

		for i in range(RAIN_DIFFUSE):
			self.rainfall = diffuse(self.rainfall, diff=0.5, decay=1.0)

		for i in range(100):
			self.temperature = diffuse(self.temperature, diff=0.5, decay=1.0)

		self.rainfall = self.rainfall / np.max(self.rainfall)

	def set_sunlight(self):
		self.sunlight = np.zeros(self.elevation.shape)
		max_dist = max(self.equator, self.height - self.equator)

		for y in range(self.height):
			row = np.ones((1, self.width)) * abs(self.equator - y) / max_dist
			self.sunlight[y,:] = row
		self.sunlight = 1 - self.sunlight

	def build_currents(self):
		border = 1
		self.currents = (np.random.random((self.height,self.width*border,2)) - 0.5) / 5

		self.equator = r.randint(int(self.height / 4), int(3 * self.height / 4))

		self.spin = r.choice([-1, 1])

		self.water = np.ones((self.height, self.width*border))

		self.water[:, int(self.width*(border-1)/2):int(self.width*(1+border)/2)] = 1 - bin_map(self.elevation.clip(0,1))
		self.currents[:,:,0] *= self.water
		self.currents[:,:,1] *= self.water

		for i in range(FIX_AND_DIFFUSE):
			if((i + 1) % 10 == 0):
				self.force_currents()
			self.diffuse_currents(self.currents)
			self.no_land_current()
			error = self.fix_currents(self.currents)
			#if(i%200==0): print('error', error)
			self.no_land_current()

		for i in range(JUST_FIX):
			self.fix_currents(self.currents)
			self.no_land_current()

		self.currents = self.currents[:, int(self.width*(border-1)/2):int(self.width*(1+border)/2)]

	def build_atmosphere(self):
		self.air_currents = self.currents[:]

		for i in range(AIR_FIX_AND_DIFFUSE):
			self.diffuse_currents(self.air_currents)
			error = self.fix_currents(self.air_currents)

		for i in range(AIR_JUST_FIX):
			self.fix_currents(self.air_currents)

	def no_land_current(self):
		self.currents[:,:,0] *= self.water
		self.currents[:,:,1] *= self.water

	def force_currents(self):
		self.currents[self.equator, :, 0] = self.spin
		self.currents[0, :, 0] = -self.spin
		self.currents[-1, :, 0] = -self.spin
		self.currents[:self.equator, 0, 1] = self.spin
		self.currents[:self.equator, -1, 1] = -self.spin
		self.currents[self.equator:, 0, 1] = -self.spin
		self.currents[self.equator:, -1, 1] = self.spin

	def diffuse_currents(self, arr):
		arr[:,:,0]=diffuse(arr[:,:,0],diff=0.3, decay=1)
		arr[:,:,1]=diffuse(arr[:,:,1],diff=0.3, decay=1)

	def fix_currents(self, arr):
		#in - out
		dx = arr[:,:-1,0] - arr[:,1:,0]
		dy = arr[:-1,:,1] - arr[1:,:,1]

		#in - out
		error = dy[:,1:] + dx[1:,:]

		delta = np.zeros(arr.shape)
		#out flow
		delta[1:,1:,0] += error
		delta[1:,1:,1] += error
		#in flow
		delta[:-1,1:,1] -= error
		delta[1:,:-1,0] -= error

		lr = 0.25
		arr += delta * lr
		return np.sum(np.abs(error))

	def make_slope_vectors(self, arr):
		slopes = np.zeros((self.height, self.width, 2))
		for y in range(1,self.height-1):
			for x in range(1, self.width-1):
				if(arr[y,x+1] < arr[y,x-1]):
					slopes[y,x,0] = arr[y,x-1] - arr[y,x]
				else:
					slopes[y,x,0] = arr[y,x] - arr[y,x+1]
				if(arr[y+1,x] < arr[y-1,x]):
					slopes[y,x,1] = arr[y-1,x] - arr[y,x]
				else:
					slopes[y,x,1] = arr[y,x] - arr[y+1,x]

		return slopes
		#return np.dstack([slope_x, slope_y])

	def crop(self, arr):
		mask = np.zeros((self.height, self.width))

		for x in range(self.width):
			for y in range(self.height):
				border = 0.15
				val1 = y / self.height / border
				val2 = x / self.width / border
				val3 = (self.height - y) / self.height / border
				val4 = (self.width - x) / self.width / border
				val5 = 1

				mask[y, x] = min(val1, val2, val3, val4, val5)

		arr = ((arr+0.8) * mask)-0.8
		arr = arr / np.max(arr)
		return arr

	def noise_map(self):
		nm = SimplexNoise()

		m_dim = max(self.height, self.width)
		pic = []
		for i in range(self.height):
			row = []
			for j in range(self.width):
					y, x = i / m_dim, j / m_dim
					noise_val = nm.noise2(self.seed + x*2,y*2) * CON_MULT
					noise_val += nm.noise2(self.seed + x*6,y*6) * REG_MULT
					noise_val += nm.noise2(self.seed + x*16,y*16) * LOC_MULT
					noise_val += nm.noise2(self.seed + x*32,y*32) * TER_MULT

					row.append(noise_val)
			pic.append(row)

		pic = (np.array(pic))
		pic = (pic / np.max(pic))


		return pic.clip(-1, 1)