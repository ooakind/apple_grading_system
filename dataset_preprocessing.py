import os
import cv2

neg_dataset_dir = 'neg_colored/'
neg_dir = 'neg/'
pos_img_file = 'pos_img/disease3_origin.jpg'

def gen_pos():
	pos_img = cv2.imread(pos_img_file, cv2.IMREAD_GRAYSCALE)
	resized_pos_img = cv2.resize(pos_img, (50, 50))
	cv2.imwrite('disease3.jpg', resized_pos_img)

def gen_neg():
	if not os.path.exists(neg_dir):
		os.makedirs(neg_dir)

	for i, file_name in enumerate(os.listdir(neg_dataset_dir)):
		neg_img = cv2.imread(neg_dataset_dir + file_name, cv2.IMREAD_GRAYSCALE)
		resized_neg_img = cv2.resize(neg_img, (100, 100))
		cv2.imwrite(neg_dir + str(i) + '.jpg', resized_neg_img)

		# os.rename(neg_dataset_dir + file_name, neg_dataset_dir + str(i) + '.jpg')


def create_descriptor():
	for dataset_type in ['neg']:
		file_names = os.listdir(dataset_type)
		for file_name in file_names:
			line = dataset_type + '/' + file_name + '\n'
			with open('bg.txt', 'a') as f:
				f.write(line)


# gen_pos()
# gen_neg()
create_descriptor()
