import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import cv2


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


def main():
    #basics_func()
    open_image()

if __name__ == '__main__':
    main()