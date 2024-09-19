import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import cv2
from numpy.ma.core import shape


def basics_func():
    pic = Image.open("DATA/00-puppy.jpg")
    pic_arr = np.asarray(pic)
    #--------------------------
    #pic.show()
    #print(pic_arr)
    #print(pic_arr.shape)
    #plt.imshow(pic_arr)
    #plt.show()
    #--------------------------
    
    #--------------------------
    #pic_red = pic_arr.copy()
    #plt.imshow(pic_red[:,:,0],cmap='gray')
    #plt.show()
    #--------------------------

    #--------------------------
    #pic_red = pic_arr.copy()
    #plt.imshow(pic_red[:,:,1],cmap='gray')
    #plt.show()
    #--------------------------

    #--------------------------
    #pic_red = pic_arr.copy()
    #pic_red[:,:,1] = 0 # set the green channel to 0
    #pic_red[:,:,0] = 0 # set the red channel to 0
    #plt.imshow(pic_red)
    #plt.show()
    #--------------------------

    #--------------------------
    #arr = np.ones((5,5))
    #print(arr * 255)
    #np.random.seed(101)
    #arr = np.random.randint(low=0, high=100,size=(5,5))
    #print(arr)
    #print(arr.max())
    #print(arr.min())
    #--------------------------

def show(img):
    plt.imshow(img)
    plt.show()

def open_image():
    img = cv2.imread("DATA/00-puppy.jpg")
    #img2 = cv2.imread("DATA/incorrect-path.jpg")
    #plt.imshow(img)
    #plt.show()
    
    # opencv .> BGR
    # matplotlib -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #plt.imshow(img_rgb)
    #plt.show()

    #img_gray = cv2.imread("DATA/00-puppy.jpg", cv2.IMREAD_GRAYSCALE)
    #plt.imshow(img_gray, cmap="gray")
    #plt.show()

    img = cv2.resize(img_rgb, (1300, 275))
    #show(img)
    w_ratio = 0.5
    h_ratio = 0.5

    new_img = cv2.resize(img_rgb, (0,0), img_rgb, w_ratio, h_ratio)
    #show(new_img)

    #new_img = cv2.flip(new_img, 0)
    #show(new_img)
    #new_img = cv2.flip(new_img, 1)
    #show(new_img)
    #new_img = cv2.flip(new_img, -1)
    #show(new_img)
    #cv2.imwrite("new_img.jpg", new_img)

    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    ax.imshow(new_img)
    plt.show()

def draw_geometry():
    blank_img = np.zeros(shape=(512,512,3), dtype=np.int16)
    print(blank_img.shape)
    #show(blank_img)
    blank_img = cv2.rectangle(blank_img, pt1=(390,10), pt2=(500,150), color=(0,255,0), thickness=5)
    #show(blank_img)
    blank_img=cv2.rectangle(blank_img, pt1=(200,200), pt2=(300,300), color=(0,0,255), thickness=5)
    #show(blank_img)
    blank_img = cv2.circle(blank_img, center=(100,100),radius=50, color=(255,0,0),thickness=10)
    #show(blank_img)
    blank_img = cv2.line(blank_img, pt1=(0,0),pt2=(511,511), color=(102,255,255),thickness=10)
    show(blank_img)

def draw_polygon():
    blank_img = np.zeros(shape=(512, 512, 3), dtype=np.int16)
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_img = cv2.putText(blank_img, text="hallo", org=(10,500), fontFace=font, fontScale=4, color=(255,255,255), thickness=3, lineType=cv2.LINE_AA)
    #show(blank_img)
    img = np.zeros(shape=(512,512,3), dtype=np.int32)
    #show(img)
    vertices = np.array([[100,300],[200,200],[400,300],[200,400]], np.int32)
    pts = vertices.reshape((-1,1,2))
    img = cv2.polylines(img,[pts],isClosed=True, color=(255,0,0), thickness=10)
    show(img)

img = np.zeros((512,512,3), np.int8)

def draw_circle(event, x, y, flags, param):
    global img
    if event == cv2.EVENT_LBUTTONDOWN:
        img = cv2.circle(img, (x,y), 100, (0,255,0), -1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        img = cv2.circle(img, (x, y), 100, (255, 0, 0), -1)

def draw_with_mouse():
    cv2.namedWindow(winname="my_drawing")
    cv2.setMouseCallback("my_drawing", draw_circle)

    while True:
        cv2.imshow('my_drawing', img)
        if cv2.waitKey(20) & 0xFF == 27:
            break

def main():
    #basics_func()
    #open_image()
    #draw_geometry()
    #draw_polygon()
    draw_with_mouse()

if __name__ == '__main__':
    main()