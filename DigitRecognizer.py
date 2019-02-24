#Import libraries
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model
import os
from tkinter import *
from PIL import ImageTk
from PIL import Image
import tkinter.messagebox
import warnings
warnings.filterwarnings("ignore")
#Set up working directory
os.chdir('C:\\Users\\Santosh Selvaraj\\Documents\\Working Directory\\Data Science Projects\\MNIST')
#Load Model
model = load_model("CNNModel.h5")

#Check all Mouse Events possible
#def MouseEvents():
#    events = [i for i in dir(cv2) if 'EVENT' in i]
#    print(events)

#Define Global Parameters
#windowName = "Draw your Digit!"
img = np.zeros((256,256,3), dtype = np.uint8)
drawing = False
(ix,iy) = (-1,-1)
output = 0
    
def drawShape(event,x,y,flags,param):
    global drawing, ix, iy    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        (ix,iy) = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),8,(255,255,255),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def drawimage():
    windowName = "Draw Digit!"
    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName,drawShape)
    while(True):
        cv2.imshow(windowName,img)
        k = cv2.waitKey(1)
        if k==27:
            break
    cv2.destroyAllWindows()

def finalImage():
    drawimage()
    global img,output
    img = img/255.0
    img = img.mean(axis=2)
    img = cv2.resize(img,(28,28))
    img = np.reshape(img,(1,28,28,1))
    results = model.predict(img)
    results = np.argmax(results, axis = 1)
    img = np.zeros((256,256,3), dtype = np.uint8)
    output = results[0].astype(np.int)

def OnClick():
    global output
    title = "Digit Prediction"
    msg = "The digit drawn was " + str(output)
    tkinter.messagebox.showinfo(title,msg)

##Tkinter
# initialize the window toolkit along with the two image panels
root = Tk()
root.title("Digit Recognizer!")
root.geometry("270x100")
root['bg'] = "gray"
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn1 = Button(root, text="Draw the Digit Image", command=finalImage, bg = "light blue")
btn1.place(relx = 0.05, rely= 0.5)
btn2 = Button(root, text="Get Prediction", command=OnClick, bg = "light green")
btn2.place(relx = 0.55, rely= 0.5)
#btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
 
# kick off the GUI
root.mainloop()