import cv2
import pytesseract
from pyzbar import pyzbar

import csv
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.1.0/bin/tesseract'

#img = cv2.imread('OCR_serial number.jpeg') # load image
img = cv2.imread('edd.png') # load image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert image to RGB

decoded = pyzbar.decode(img)
print(decoded)

# # Detecting characters
# hImg, wImg,_ = img.shape
# #conf = r'--oem 3 --psm 6 outputbase digits'
# boxes = pytesseract.image_to_boxes(img) # character bounding boxes, x y width height
#
#
# for b in boxes.splitlines():
#     #print(b)
#     b = b.split(' ') #split each value based on space [ch,x,y,w,h]
#     print(b)
#     x,y,w,h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
#     cv2.rectangle(img,(x,hImg-y), (w,hImg-h), (0,0,255),2)
#     cv2.putText(img, b[0],(x,hImg-y+25), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255),1)
#
#
#
#
# cv2.imshow('Result', img) #show image
# cv2.waitKey(0)
