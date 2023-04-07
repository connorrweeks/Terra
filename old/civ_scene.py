#from scene import *
#import scene
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

from terrain_maker import TerrainMaker, seed
from civ_map import CivMap
from map_drawer import MapDrawer

NUM_VIEWS = 9
view_titles = ['biome mode', 'island mode', 'simple mode', 'weather mode', 'rainfall mode', 'temperature mode',
               'political mode', 'normal mode', 'occupied mode']

MAP_NAME = ''


class MyScene(Scene):
    def __init__(self):
        super().__init__()
        f = open('./log.txt', 'w+')
        f.close()

        self.display = None
        self.v_mode = 6
        self.md = MapDrawer(self.size.w, self.size.h, self.v_mode)
        self.offset = Vector2()
        self.map_scale = 1.0
        self.image_scale = 1.0
        self.zoom = False
        self.paused = False

    def setup(self):
        try:
            self.real_setup()
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            assert (False)

    def real_setup(self):
        # n = r.randrange(40, 50)
        # n = 15
        self.map = CivMap(self.size.w, self.size.h)
        seed(2)
        # self.map.generate()
        # self.map.save_pickle('2')
        self.map.load_pickle('2')

        # for i in range(10):
        #		self.map.make_a_city()

        # assert(False)
        self.make_img()
        self.update_display()
        # self.display_new_world()
        self.start = time.perf_counter()
        # self.map_counter = 0
        # self.display_world('15')

        self.title = LabelNode(view_titles[self.v_mode])
        self.title.position = Vector2(self.size.w / 2, self.size.h - 50)
        self.title.z_position = 1
        self.add_child(self.title)

        c_height = self.size.h / 12
        c_width = self.size.w / 3
        shape = ui.Path.rect(0, 0, c_width, c_height)
        shape.line_width = 3
        self.view_button = ShapeNode(shape, position=Vector2(c_width / 2, c_height / 2), fill_color='green',
                                     stroke_color='black')
        self.view_button.z_position = 1
        self.add_child(self.view_button)

        self.v_button2 = ShapeNode(shape, position=Vector2(c_width * 1.5, c_height / 2), fill_color='green',
                                   stroke_color='black')
        self.v_button2.z_position = 1
        self.add_child(self.v_button2)

        shape = ui.Path.rect(0, 0, c_width, c_height)
        shape.line_width = 3
        self.pause_button = ShapeNode(shape, position=Vector2(c_width * 2.5, c_height / 2), fill_color='gray',
                                      stroke_color='black')
        self.pause_button.z_position = 1
        self.add_child(self.pause_button)

        v_mode_icon = SpriteNode('iow:ios7_refresh_empty_256', position=Vector2(0, 0), scale=0.5)
        v_mode_icon.size = Vector2(c_height * 1.5, c_height * 1.5)
        v_mode_icon.position = Vector2(c_width / 2, c_height / 2)
        v_mode_icon.z_position = 3
        self.add_child(v_mode_icon)

        pause_icon = SpriteNode('iow:ios7_pause_256', position=Vector2(0, 0), scale=0.5)
        pause_icon.size = Vector2(c_height * 1.5, c_height * 1.5)
        pause_icon.position = Vector2(c_width * 2.5, c_height / 2)
        pause_icon.z_position = 3
        self.add_child(pause_icon)

        self.turn_speed = 0.1

    def update(self):
        if (self.start < time.perf_counter() and not self.paused):
            self.start += self.turn_speed
            t0 = time.perf_counter()
            self.map.full_turn()
            for c in self.map.all_civs:
                if (c.level == 2):
                    self.turn_speed = 0.5
            self.make_img()
            self.display.run_action(Action.remove())
            self.display = None
            self.update_display()
            print(f'turn time: {time.perf_counter() - t0:.2f}')

    def display_world(self, n):
        self.tm = TerrainMaker(self.size.w / 2, self.size.h / 2)
        # self.tm.load_pickle(n)
        tm.generate()
        tm.save_pickle(n)

        self.make_img()
        self.update_display()

    def make_img(self, n=''):
        # md = MapDrawer(self.size.w, self.size.h, self.v_mode)

        cm = self.md.to_colors(self.map)
        # if(self.display != None):

        # left_chop =

        # show_midx = (-self.display.position.x / self.map_scale)

        # take_height = min(self.size.h, self.size.h / self.map_scale)
        # take_width = min(self.size.w, self.size.w / self.map_scale)

        # take_left = cm.shape[1] - (self.display.position.x / self.map_scale)
        # take_top = cm.shape[0] - (self.display.position.y /self.map_scale)

        # self.take_left = take_left
        # self.take_top = take_top
        # take_left = (self.display.position.x / self.map_scale)
        # take_top = (self.display.position.y / self.map_scale)
        # pass
        # cm = cm[max(0, int(take_top-(take_height/2))):min(cm.shape[0]-1,int(take_top+(take_height/2))),max(0,int(take_left-(take_width/2))):min(cm.shape[1]-1,int(take_left+(take_width/2))), :]

        # print(int(take_top-(take_height/2)), '=', int(take_top+(take_height/2)))
        # print(cm.shape, take_left, take_width, take_top, take_height)
        im = self.md.to_image(cm)
        if (n != ''):
            im.save(f"./maps/{n}.png")
        self.img = self.md.to_ui(im)  # , self.image_scale)

    def did_change_size(self):
        pass

    def update_display(self):
        if (self.display == None):
            self.display = self.array2sprite(self.img)
            self.add_child(self.display)
        else:
            if (self.image_scale > self.map_scale):
                self.image_scale = max(1, self.image_scale / 2)
            if (self.image_scale < self.map_scale / 2):
                self.image_scale *= 2
            self.display.position = Vector2(self.size.w / 2, self.size.h / 2) - (self.offset)
        # self.display.scale = self.map_scale

    def touch_began(self, touch):
        self.last_touch = touch.location
        if (len(self.touches) == 2):
            t_arr = [self.touches[k] for k in self.touches]
            self.touch_distance = abs(t_arr[0].location - t_arr[1].location)
            self.zoom = True
        elif (self.in_button(touch.location, self.view_button)):
            self.v_mode = (self.v_mode - 1) % NUM_VIEWS
            self.md.set_view(self.v_mode)
            self.title.text = view_titles[self.v_mode]
        elif (self.in_button(touch.location, self.v_button2)):
            self.v_mode = (self.v_mode + 1) % NUM_VIEWS
            self.md.set_view(self.v_mode)
            self.title.text = view_titles[self.v_mode]
        # self.make_img()
        # self.display.run_action (Action.remove())
        # self.display = None
        # self.update_display()
        elif (self.in_button(touch.location, self.pause_button)):
            self.paused = not self.paused
            self.start = time.perf_counter()

    def in_button(self, loc, b):
        return (loc.x > b.position.x and loc.y > b.position.y and
                loc.x < b.position.x + b.size.w and loc.y < b.position.y + b.size.h)

    def touch_moved(self, touch):
        if (len(self.touches) == 2):
            t_arr = [self.touches[k] for k in self.touches]
            new_touch_distance = abs(t_arr[0].location - t_arr[1].location)
            self.map_scale *= new_touch_distance / self.touch_distance
            self.touch_distance = new_touch_distance
            self.map_scale = max(1, self.map_scale)
        elif (len(self.touches) == 1 and self.zoom == False):
            self.offset += self.last_touch - touch.location
            self.last_touch = touch.location
        self.offset.x = max((self.map_scale - 1) * self.size.w / -2, self.offset.x)
        self.offset.y = max((self.map_scale - 1) * self.size.h / -2, self.offset.y)
        self.offset.x = min((self.map_scale - 1) * self.size.w / 2, self.offset.x)
        self.offset.y = min((self.map_scale - 1) * self.size.h / 2, self.offset.y)
        self.update_display()

    def touch_ended(self, touch):
        if (len(self.touches) == 0):
            self.zoom = False

    def array2sprite(self, img):
        texture = Texture(img)
        pos = Vector2(self.size.w / 2, self.size.h / 2) - self.offset
        node = SpriteNode(texture, position=pos, scale=self.map_scale)
        return node


if __name__ == '__main__':
    run(MyScene(), show_fps=False)