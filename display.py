from terrain_maker import TerrainMaker, seed
from civ_map import CivMap
from map_drawer import MapDrawer
from PIL import ImageTk, Image
from tkinter import *
import tkinter.messagebox as msgbox
import time

global pause
pause = False
def change_mode():
    global map_mode
    map_mode = (map_mode + 1) % 5
    # msgbox.showinfo( "Hello Python", f"New mode: {map_mode}")
    #global pause
    #pause = not pause

n = 12

seed(n)


h = 400
w = 400

map_mode = 1

tm = TerrainMaker(w, h)
md = MapDrawer(w, h)
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

img = ImageTk.PhotoImage(im)
label = Label(canvas,image=img)
label.pack(side=LEFT)

control_panel = Frame(canvas)
control_panel.pack()

Button(control_panel,text="change mode",command=change_mode).pack()
Button(control_panel, text ="phase1", command = tm.phase1).pack()
Button(control_panel, text ="phase2", command = tm.phase2).pack()
Button(control_panel, text ="phase3", command = tm.phase3).pack()
Button(control_panel, text ="phase4", command = tm.phase4).pack()
Button(control_panel, text ="phase5", command = tm.phase5).pack()
Button(control_panel, text ="phase6", command = tm.phase6).pack()
Button(control_panel, text ="save map", command = save_map).pack()

refresh_rate = 200

def message():
    global md
    md.set_view_int(map_mode)
    global pause
    if(pause == False):
        cm = md.to_colors(tm)
        im = md.to_image(cm)
        im = im.resize((w*2, h*2))
        img2 = ImageTk.PhotoImage(im)
        label.configure(image=img2)
        label.image = img2
    canvas.after(refresh_rate, message)

canvas.after(refresh_rate, message)
canvas.mainloop()

from tkinter import *
from PIL import ImageTk, Image

#im.show()
#if (n != ''):
#    im.save(f"./maps/{n}.png")