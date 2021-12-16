import os
import cv2
import time
import matplotlib.image as pimg
import matplotlib.pyplot as plt

disease_cascade = cv2.CascadeClassifier('data/cascade.xml')
test_data_path = 'input/'
result_data_path = 'result/'

if not os.path.exists(result_data_path):
	os.makedirs(result_data_path)

for f in os.listdir(test_data_path):
	print(f)
	disease_num = 0
	start = time.time()
	img = cv2.imread(test_data_path + f)
	resized_img = cv2.resize(img, (210, 100))
	gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
	disease_cascade_list = disease_cascade.detectMultiScale(gray, 50, 50)

	for (x, y, w, h) in disease_cascade_list:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
		disease_num += 1

	cv2.imwrite(result_data_path + 'detect_'+ f, img)

	print(f'Number of diseases: {disease_num}')
	print(f'Grading time: {time.time() - start} sec\n')



