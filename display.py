from terrain_maker import TerrainMaker, seed
from civ_map import CivMap
from map_drawer import MapDrawer
from PIL import ImageTk, Image
from tkinter import *
import tkinter.messagebox as msgbox
import time
import cv2

global pause
pause = False
def change_mode():
    global map_mode
    map_mode = (map_mode + 1) % 6
    # msgbox.showinfo( "Hello Python", f"New mode: {map_mode}")
    #global pause
    #pause = not pause

def change_map():
    global tm
    global n
    global md
    n = (n + 1) % 30
    tm.load_pickle(n)
    md = MapDrawer(w, h)

n = 12

h = 400
w = 400

CONSTANT_SCALER = int(1000 / h)

seed(n)

map_mode = 1

tm = TerrainMaker()
md = MapDrawer()
tm.seed = n
tm.t0 = time.perf_counter()
#tm.phase1()

#tm.generate()
#tm.save_pickle(n)

#map = CivMap(w, h)

tm.load_pickle(n)

def save_map():
    tm.save_pickle(n)

cm = md.to_colors(tm)
im = md.to_image(cm)

canvas = Canvas(Tk(), width=w*2+100, height=h*2+100, bg="#000000")
canvas.pack()

mouse_down = False



img = ImageTk.PhotoImage(im)
label = Button(canvas,image=img)
label.pack(side=LEFT)



control_panel = Frame(canvas)
control_panel.pack()

class MouseControl():
    def __init__(self, h, w, button):
        self.button = button
        button.bind("<Button 1>",self.touch_began)
        button.bind("<ButtonRelease 1>",self.touch_ends)
        button.bind("<Double-Button 1>",self.zoom_in)
        button.bind("<Double-Button 3>",self.zoom_out)
        button.bind("<Leave>",self.touch_ends)
        button.bind("<Motion>",self.touch_moves)
        self.mouse_down = False
        self.zoom_level = 1
        self.x = 0
        self.y = 0
        self.h = h
        self.w = w

        self.zoom_width = w
        self.zoom_height = h

        self.touch_x = 0
        self.touch_y = 0

        self.t0 = -100

        self.zoom_speed = 1.2
        self.cm = None

        self.refresh_rate = 50
        self.regen_rate = 1000

    def touch_moves(self, touch):
        if(self.mouse_down == False): return
        diff_x = int(touch.x / CONSTANT_SCALER) - self.touch_x
        diff_y = int(touch.y / CONSTANT_SCALER) - self.touch_y

        self.touch_x = int(touch.x / CONSTANT_SCALER)
        self.touch_y = int(touch.y / CONSTANT_SCALER)
        
        self.x = self.x - (diff_x / self.zoom_level)
        self.y = self.y - (diff_y / self.zoom_level)
        self.x = int(min(max(0, self.x), self.w - (self.w / self.zoom_level)))
        self.y = int(min(max(0, self.y), self.h - (self.h / self.zoom_level)))

    def touch_began(self, touch):
        self.mouse_down = True
        self.touch_x = int(touch.x / CONSTANT_SCALER * self.zoom_level)
        self.touch_y = int(touch.y / CONSTANT_SCALER * self.zoom_level)
        
    def touch_ends(self, touch):
        print("Mouse up")
        self.mouse_down = False

    def zoom_in(self, touch):
        print("ZOOM IN")
        self.zoom_level = self.zoom_level * self.zoom_speed
        self.zoom_level = min(self.zoom_level, 10)
        print(self.zoom_level)

    def zoom_out(self, touch):
        print("ZOOM OUT")
        self.zoom_level = self.zoom_level / self.zoom_speed
        self.zoom_level = max(self.zoom_level, 1)
        self.x = int(min(max(0, self.x), self.w - (self.w / self.zoom_level)))
        self.y = int(min(max(0, self.y), self.h - (self.h / self.zoom_level)))

    def message(self):
        global md
        md.set_view_int(map_mode)
        global pause
        
        if(pause == False):
            print(time.perf_counter())
            if(time.perf_counter() > self.t0 + (self.regen_rate / 1000.0) or self.cm is None):
                print("Regen")
                self.cm = md.to_colors(tm)
                self.t0 = time.perf_counter()
            self.zoom_cm = self.cm[self.y:self.y+int(self.h/self.zoom_level),self.x:self.x+int(self.w/self.zoom_level),:]
            im = md.to_image(self.zoom_cm)
            im = im.resize((w*CONSTANT_SCALER, h*CONSTANT_SCALER))
            img2 = ImageTk.PhotoImage(im)
            label.configure(image=img2)
            label.image = img2
        canvas.after(self.refresh_rate, self.message)
MC = MouseControl(h, w, label)

Button(control_panel,text="change mode",command=change_mode).pack()
Button(control_panel, text ="phase1", command = tm.phase1).pack()
Button(control_panel, text ="phase2", command = tm.phase2).pack()
Button(control_panel, text ="phase3", command = tm.phase3).pack()
Button(control_panel, text ="phase4", command = tm.phase4).pack()
Button(control_panel, text ="phase5", command = tm.phase5).pack()
Button(control_panel, text ="phase6", command = tm.phase6).pack()
Button(control_panel, text ="save map", command = save_map).pack()
Button(control_panel, text ="change map", command = change_map).pack()



canvas.after(10, MC.message)
canvas.mainloop()

from tkinter import *
from PIL import ImageTk, Image

#im.show()
#if (n != ''):
#    im.save(f"./maps/{n}.png")