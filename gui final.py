# !/usr/bin/python3
from tkinter import *
import sys
import os


from tkinter import messagebox

top = Tk()
top.geometry("600x600")
def helloCallBack():
     
    print("Election commsion")
    #sys.cmd('python fall_detection_final.py')
    os.system('python final_evm2.py')
   

B = Button(top, text = "start", command = helloCallBack)
B.place(x = 300,y = 300)
top.mainloop()

