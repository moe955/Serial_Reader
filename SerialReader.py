import cv2
import pytesseract
from pyzbar import pyzbar

import csv
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.1.0/bin/tesseract'

img = cv2.imread('edd.png') # load image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert image to RGB

decoded = pyzbar.decode(img)
print(decoded)
