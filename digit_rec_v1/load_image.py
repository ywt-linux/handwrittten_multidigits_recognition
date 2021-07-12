'''
Author: ywt
Date: 2021-07-11 11:13:44
LastEditTime: 2021-07-12 10:38:06
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /digit_rec_v1/load_image.py
'''
import cv2
import numpy as np
from PIL import Image
from imutils import contours

img = cv2.imread('test_fig/no_order.png')
lowerb = (0,0,116)
upperb = (255,255,255)
back_mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lowerb, upperb)
cv2.imwrite('generated_fig/fig2.jpg', back_mask)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
back_mask = cv2.erode(back_mask, kernel, iterations=1)

num_mask = cv2.bitwise_not(back_mask)

num_mask = cv2.medianBlur(num_mask,3) 
cv2.imwrite('generated_fig/fig3.jpg', num_mask)


# find contours
contours, hier = cv2.findContours(num_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

canvas = cv2.cvtColor(num_mask, cv2.COLOR_GRAY2BGR)
def getStandardDigit(img):
    STD_WIDTH = 32 
    STD_HEIGHT = 64
    height,width = img.shape
    
    new_width = int(width * STD_HEIGHT / height)
    if new_width > STD_WIDTH:
        new_width = STD_WIDTH
    
    resized_num = cv2.resize(img, (new_width,STD_HEIGHT), interpolation = cv2.INTER_NEAREST)
    
    canvas = np.zeros((STD_HEIGHT, STD_WIDTH))
    x = int((STD_WIDTH - new_width) / 2) 
    canvas[:, x:x+new_width] = resized_num
    return canvas
 
minWidth = 5 
minHeight = 20
 
imgIdx = 0 # count

axis0 = []
ord = []
for cidx,cnt in enumerate(contours):
    (x, y, w, h) = cv2.boundingRect(cnt)
    # delete contour smaller than minwidth and minheight
    if w < minWidth or h < minHeight:
        continue
    # get ROI fig
    digit = num_mask[y:y+h, x:x+w]
    digit = getStandardDigit(digit)
    ord.append(digit)
    axis0.append(x)
    imgIdx+=1
    # plot
    cv2.rectangle(canvas, pt1=(x, y), pt2=(x+w, y+h),color=(0, 255, 255), thickness=2) 
    print('X:', x, '\n', 'Y:', y) 
cv2.imwrite('after_seperate.png', canvas)

# sort(from left to right)
order = np.argsort(axis0)
print(axis0)
k = 0
for i in order:
    cv2.imwrite('generated_fig/{}.png'.format(k), ord[i])
    k += 1
print(order)
img28 = np.zeros([28, 28], np.uint8)
cv2.imwrite('img28.png', img28)
numbers = []


for i in range(imgIdx):
    num_mask = cv2.imread('generated_fig/{}.png'.format(i), 0)
    num_mask = cv2.resize(num_mask, (20, 20))
    cv2.imwrite('generated_fig/{}.png'.format(i), num_mask)
    img28 = Image.open('img28.png')
    num_mask = Image.open('generated_fig/{}.png'.format(i))
    img28.paste(num_mask,(4, 4))
    img28 = cv2.cvtColor(np.asarray(img28), cv2.COLOR_RGB2BGR)  
    cv2.imwrite('generated_fig/{}.png'.format(i), img28)
    img28 = cv2.cvtColor(np.asarray(img28), cv2.COLOR_BGR2GRAY)  
    numbers.append(img28)


numbers = np.array(numbers)
print(numbers.shape)
np.save('data/data', numbers)

print('successfully saved')