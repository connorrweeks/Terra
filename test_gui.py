from tkinter import Tk, Canvas, Button, Label, PhotoImage, mainloop
import random

WIDTH, HEIGHT = 200, 200

def modify_image():
  print ("modifiying image...")
  for x in range(1000):
    img.put ( '#%06x' % random.randint(0, 16777215),   # 6 char hex color
      ( random.randint(0, WIDTH), random.randint(0, HEIGHT) )   # (x, y)
    )
  canvas.update_idletasks()
  print ("done")

canvas = Canvas(Tk(), width=WIDTH, height=HEIGHT, bg="#000000")
canvas.pack()

Button(canvas,text="modifiying image",command=modify_image).pack()
img = PhotoImage(width=WIDTH, height=HEIGHT)
Label(canvas,image=img).pack()
mainloop()