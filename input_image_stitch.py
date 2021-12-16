import sys
# if 'google.colab' in sys.modules:
#     import subprocess
#     subprocess.call("pip install -U opencv-python".split())

# from google.colab import drive
# drive.mount('/content/drive')

# from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import cv2
import os
import glob
import numpy as np


img_dir = 'stitch_input/'
result_dir = 'input/'


if not os.path.exists(img_dir):
	os.makedirs(img_dir)
if not os.path.exists(result_dir):
	os.makedirs(result_dir)

img_bgr0 = cv2.imread(img_dir + 'apple0.jpg', cv2.IMREAD_COLOR)
img_bgr0 = cv2.resize(img_bgr0, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
img_bgr1 = cv2.imread(img_dir + 'apple12.jpg', cv2.IMREAD_COLOR)
img_bgr1 = cv2.resize(img_bgr1, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
img_bgr2 = cv2.imread(img_dir + 'apple2.jpg', cv2.IMREAD_COLOR)
img_bgr2 = cv2.resize(img_bgr2, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
img_bgr3 = cv2.imread(img_dir + 'apple3.jpg', cv2.IMREAD_COLOR)
img_bgr3 = cv2.resize(img_bgr3, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

# def pause():
#     # pause
#     keycode = cv2.waitKey(0)
#     # ESC key to close imshow
#     if keycode == 27:
#         cv2.destroyAllWindows()


# cv2_imshow(img_bgr)
# pause()

img_bitwise_not_bgr = cv2.bitwise_not(img_bgr0)
# cv2_imshow(img_bitwise_not_bgr)
# pause()

img_bitwise_not_bgr2gray = cv2.cvtColor(img_bitwise_not_bgr, cv2.COLOR_BGR2GRAY)
# cv2_imshow(img_bitwise_not_bgr2gray)
# pause()

ret, img_binary = cv2.threshold(img_bitwise_not_bgr2gray, 150, 180, cv2.THRESH_BINARY)
# cv2_imshow(img_binary)
# pause()

contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# contours 는 튜플형. ([[[x1,y1]],[[x2,y2]],[[x3,y3]]])
contours = list(contours)
print(len(contours))
contours = contours[3]

x_vals = []
y_vals = []
left_con = []
right_con = []

for i in range(len(contours)):
    x_vals.append(contours[i][0][0])
    y_vals.append(contours[i][0][1])

x_max = max(x_vals)
x_min = min(x_vals)
y_max = max(y_vals)
y_min = min(y_vals)

x_avg = (x_max + x_min)/2

for i in range(len(contours)):
    if contours[i][0][0] <= x_avg :
        left_con.append(contours[i].tolist())
    else :
        right_con.append(contours[i].tolist())

x_vals_2 = []
for i in range(len(left_con)):
    x = left_con[i][0][0]
    x = int((x_avg + x)/2)
    x_vals_2.append(x)
    for j in range(x):
        img_bgr0[left_con[i][0][1],j] = (0,0,0)


for i in range(len(right_con)):
    x = right_con[i][0][0]
    x = int((x_avg + x)/2)
    x_vals_2.append(x)
    for j in range(img_bgr0.shape[1]-x):
        img_bgr0[right_con[i][0][1],x+j] = (0,0,0)

x_min = min(x_vals_2)
x_max = max(x_vals_2)
img_bgr0 = img_bgr0[y_min:y_max,x_min:x_max]


img_bitwise_not_bgr = cv2.bitwise_not(img_bgr1)
img_bitwise_not_bgr2gray = cv2.cvtColor(img_bitwise_not_bgr, cv2.COLOR_BGR2GRAY)
ret, img_binary = cv2.threshold(img_bitwise_not_bgr2gray, 150, 180, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# contours 는 튜플형. ([[[x1,y1]],[[x2,y2]],[[x3,y3]]])
contours = contours[3]
x_vals = []
y_vals = []
left_con = []
right_con = []
for i in range(len(contours)):
    x_vals.append(contours[i][0][0])
    y_vals.append(contours[i][0][1])

x_max = max(x_vals)
x_min = min(x_vals)
y_max = max(y_vals)
y_min = min(y_vals)

x_avg = (x_max + x_min)/2

for i in range(len(contours)):
    if contours[i][0][0] <= x_avg :
        left_con.append(contours[i].tolist())
    else :
        right_con.append(contours[i].tolist())

x_vals_2 = []
for i in range(len(left_con)):
    x = left_con[i][0][0]
    x = int((x_avg + x)/2)
    x_vals_2.append(x)
    for j in range(x):
        img_bgr1[left_con[i][0][1],j] = (0,0,0)

for i in range(len(right_con)):
    x = right_con[i][0][0]
    x = int((x_avg + x)/2)
    x_vals_2.append(x)
    for j in range(img_bgr1.shape[1]-x):
        img_bgr1[right_con[i][0][1],x+j] = (0,0,0)
x_min = min(x_vals_2)
x_max = max(x_vals_2)
img_bgr1 = img_bgr1[y_min:y_max,x_min:x_max]

img_bitwise_not_bgr = cv2.bitwise_not(img_bgr2)
img_bitwise_not_bgr2gray = cv2.cvtColor(img_bitwise_not_bgr, cv2.COLOR_BGR2GRAY)
ret, img_binary = cv2.threshold(img_bitwise_not_bgr2gray, 150, 180, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# contours 는 튜플형. ([[[x1,y1]],[[x2,y2]],[[x3,y3]]])
contours = contours[8]
x_vals = []
y_vals = []
left_con = []
right_con = []
for i in range(len(contours)):
    x_vals.append(contours[i][0][0])
    y_vals.append(contours[i][0][1])

x_max = max(x_vals)
x_min = min(x_vals)
y_max = max(y_vals)
y_min = min(y_vals)

x_avg = (x_max + x_min)/2

for i in range(len(contours)):
    if contours[i][0][0] <= x_avg :
        left_con.append(contours[i].tolist())
    else :
        right_con.append(contours[i].tolist())

x_vals_2 = []
for i in range(len(left_con)):
    x = left_con[i][0][0]
    x = int((x_avg + x)/2)
    x_vals_2.append(x)
    for j in range(x):
        img_bgr2[left_con[i][0][1],j] = (0,0,0)

for i in range(len(right_con)):
    x = right_con[i][0][0]
    x = int((x_avg + x)/2)
    x_vals_2.append(x)
    for j in range(img_bgr2.shape[1]-x):
        img_bgr2[right_con[i][0][1],x+j] = (0,0,0)
x_min = min(x_vals_2)
x_max = max(x_vals_2)
print(y_min,y_max,x_min,x_max)
img_bgr2 = img_bgr2[y_min:y_max,x_min:x_max]

img_bitwise_not_bgr = cv2.bitwise_not(img_bgr3)
img_bitwise_not_bgr2gray = cv2.cvtColor(img_bitwise_not_bgr, cv2.COLOR_BGR2GRAY)
ret, img_binary = cv2.threshold(img_bitwise_not_bgr2gray, 150, 180, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# contours 는 튜플형. ([[[x1,y1]],[[x2,y2]],[[x3,y3]]])
contours = contours[0]
x_vals = []
y_vals = []
left_con = []
right_con = []
for i in range(len(contours)):
    x_vals.append(contours[i][0][0])
    y_vals.append(contours[i][0][1])

x_max = max(x_vals)
x_min = min(x_vals)
y_max = max(y_vals)
y_min = min(y_vals)

x_avg = (x_max + x_min)/2

for i in range(len(contours)):
    if contours[i][0][0] <= x_avg :
        left_con.append(contours[i].tolist())
    else :
        right_con.append(contours[i].tolist())

x_vals_2 = []
for i in range(len(left_con)):
    x = left_con[i][0][0]
    x = int((x_avg + x)/2)
    x_vals_2.append(x)
    for j in range(x):
        img_bgr3[left_con[i][0][1],j] = (0,0,0)

for i in range(len(right_con)):
    x = right_con[i][0][0]
    x = int((x_avg + x)/2)
    x_vals_2.append(x)
    for j in range(img_bgr3.shape[1]-x):
        img_bgr3[right_con[i][0][1],x+j] = (0,0,0)
x_min = min(x_vals_2)
x_max = max(x_vals_2)
img_bgr3 = img_bgr3[y_min:y_max,x_min:x_max]

imgs = [img_bgr0,img_bgr1,img_bgr2,img_bgr3]

h_min = 1000

for i in range(len(imgs)):
    h = imgs[i].shape[0]
    if h <= h_min:
        h_min = h
    rsz_ratio = h_min/h

for i in range(4):
    w = int(imgs[i].shape[1]*h_min/imgs[i].shape[0])
    imgs[i] = cv2.resize(imgs[i], dsize=(w,h_min))

addh = cv2.hconcat(imgs)
#cv2_imshow(addh)
#pause()

 cv2.imwrite(result_dir + 'result01.jpg',addh)