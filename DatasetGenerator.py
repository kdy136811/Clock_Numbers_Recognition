import numpy as np
import cv2
import random
import Mnist
import matplotlib.pyplot as plt
import os

def preProcess(img, angle, scale):
    (h,w) = img.shape[:2]
    center = (w/2,h/2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rimg = cv2.warpAffine(img, M, (h,w))
    rsimg = cv2.resize(rimg, (int(scale*w),int(scale*h)))
    nh, nw = rsimg.shape[:2]
    #print(rsimg.shape, angle, scale)

    return rsimg, nw, nh

def getPosition(side, w, h, checkmap):
    while(1):
        xpos = random.randint(0,side-h)
        ypos = random.randint(0,side-w)
        overlap, checkmap = checkOverlap(xpos,ypos,w,h,checkmap)
        if(overlap):
            break
    
    return xpos, ypos, checkmap

def checkOverlap(x, y, w, h, checkmap):
    for i in range(x, x+h):
        for j in range(y, y+w):
            if checkmap[i][j] == 1:
                return False, checkmap
    for i in range(x, x+h):
        for j in range(y, y+w):
            checkmap[i][j] = 1

    return True, checkmap

def cropImage(img, w, h, side):
    if(side=='left'):
        crop = img[0:h,0:int(0.8214*w)]
    if(side=='right'):
        crop = img[0:h,int(0.1785*w):w]

    return crop

def combineTwoNums(lnum, rnum):
    num1_size = len(number_index[lnum])
    num2_size = len(number_index[rnum])
    n1 = x_train[number_index[lnum][random.randint(0,num1_size-1)]].reshape([28,28])
    n2 = x_train[number_index[rnum][random.randint(0,num2_size-1)]].reshape([28,28])
    rotate = random.randint(-5,10)
    scale = random.uniform(1.5,3.0)
    n1, w1, h1 = preProcess(n1, rotate, scale)
    n2, w2, h2 = preProcess(n2, rotate, scale)
    n1 = cropImage(n1, w1, h1, 'left')
    n2 = cropImage(n2, w2, h2, 'right')
    combined_num = np.hstack([n1,n2])
    nh, nw = combined_num.shape[:2]

    return combined_num, nw, nh

if __name__ == "__main__":
    #Load dataset
    Mnist.downloadMNIST(path='MNIST_data', unzip=True)
    x_train, y_train = Mnist.loadMNIST(path="MNIST_data")
    data_size = y_train.shape[0]

    #Classify each number
    number_index = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(data_size):
        number_index[y_train[i]].append(i)

    if(not (os.path.isdir('clock'))):
        os.mkdir('clock')
    TOTAL_IMAGES = 10
    for image in range(TOTAL_IMAGES):
        filename = 'clock/'+ str(image).zfill(5)
        obj_list = []

        #Generate 512*512 black image
        side = 512
        background = np.zeros((side,side), np.uint8)
        check_map = np.zeros((side, side), np.int)

        #Put 1 to 9 on background image
        for i in range(1,10):
            num_size = len(number_index[i])
            num = x_train[number_index[i][random.randint(0,num_size-1)]].reshape([28,28])
            new_num, w, h = preProcess(num, random.randint(-10,15), random.uniform(1.5,3.0))
            xpos, ypos, check_map = getPosition(side, w, h, check_map)
            background[xpos:xpos+h, ypos:ypos+w] = new_num
            obj_box = [i-1, (ypos+w/2)/side, (xpos+h/2)/side, w/side, h/side]
            obj_list.append(obj_box)
        
        #Put 10-12 on background image
        for i in range(3):
            over_nine, w, h = combineTwoNums(1,i)
            xpos, ypos, checkmap = getPosition(side, w, h, check_map)
            background[xpos:xpos+h, ypos:ypos+w] = over_nine
            obj_box = [9+i, (ypos+w/2)/side, (xpos+h/2)/side, w/side, h/side]
            obj_list.append(obj_box)

        #Binarization
        ret, binary = cv2.threshold(background, 110, 255, cv2.THRESH_BINARY_INV)

        #cv2.imshow('black', binary)
        #cv2.waitKey(0)

        #Output images and YOLO labels
        cv2.imwrite(filename+'.jpg', binary)
        with open(filename+'.txt', 'w') as text_file:
            for obj in obj_list:
                print('{0} {1} {2} {3} {4}'.format(obj[0], obj[1], obj[2], obj[3], obj[4]), file=text_file)