import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import cv2
from numpy.ma.core import shape
from reportlab.lib.colors import white, gray


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

def show(img, gray=False):
    if gray:
        plt.imshow(img,cmap='gray')
    else:
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

def farb_mapping():
    img = cv2.imread("DATA/00-puppy.jpg")
    #show(img)
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    show(img_hls)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    show(img_hsv)

def bilder_mischen_und_einfügen():
    img1 = cv2.imread("DATA/dog_backpack.jpg")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread("DATA/watermark_no_copy.png")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    #show(img1)
    #show(img2)
    img1 = cv2.resize(img1, (1200,1200))
    img2 = cv2.resize(img2, (1200,1200))
    blended = cv2.addWeighted(src1=img1, alpha=0.4 , src2=img2, beta = 0.6, gamma=0)
    show(blended)

def masken():
    img1 = cv2.imread("DATA/dog_backpack.jpg")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread("DATA/watermark_no_copy.png")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # show(img1)
    # show(img2
    img2 = cv2.resize(img2, (600, 600))
    # image1.shape = 1401, 934, 3
    x_offset = 934-600
    y_offset = 1401-600
    #ROI: Region of Interest
    roi = img1[y_offset:1401, x_offset:934]
    #show(roi)
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    mask_inv = cv2.bitwise_not(img2gray)
    #show(mask_inv, True)
    white_background = np.full(img2.shape, 255, dtype=np.uint8)
    bk = cv2.bitwise_or(white_background, white_background, mask=mask_inv)
    #show(bk)
    fg = cv2.bitwise_or(img2, img2, mask=mask_inv)
    #show(fg)
    final_roi = cv2.bitwise_or(roi, fg)
    #show(final_roi)

    large_img = img1
    small_img = final_roi
    large_img[y_offset:y_offset+small_img.shape[0],x_offset:x_offset+small_img.shape[1]] = small_img
    show(large_img)

def thresholding():
    #img = cv2.imread("DATA/rainbow.jpg", 0)
    #show(img, gray=True)
    #ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    #ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    #ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    #ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    #ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
    #show(thresh1, gray=True)

    img = cv2.imread("DATA/crossword.jpg", 0)
    ret, th1 = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
    #show(th1, gray=True)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 8)
    show(th2, gray=True)
    blended = cv2.addWeighted(src1=th1, alpha=0.7, src2=th2, beta=0.3, gamma=0)
    show(blended, gray=True)

def main():
    #basics_func()
    #open_image()
    #draw_geometry()
    #draw_polygon()
    #draw_with_mouse()
    #farb_mapping()
    #bilder_mischen_und_einfügen()
    #masken()
    thresholding()

if __name__ == '__main__':
    main()