import cv2
from HistogramVisualization import plot

img_cat = cv2.imread('basic_filtering/img/cat.jpg', 0)

plot(img_cat, 'Original', 121, (16,8), True)
