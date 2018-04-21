import os
import pickle
import numpy as np
import tensorflow as tf
from datetime import datetime
from AlexNet import AlexNet
import preprocessing_rcnn as prep
import skimage
import selectivesearch
from PIL import Image
from matplotlib.font_manager import FontProperties 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import buttfly_classes

#resize the image to 227*227
def resize_image(in_image, new_width, new_height, out_image=None,
                 resize_mode=Image.ANTIALIAS):
    img = in_image.resize((new_width, new_height), resize_mode)
    if out_image:
        img.save(out_image)
    return img
#change the pixel to float32
def pil_to_nparray(pil_image):
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")
#get the regions proposal of the test image
def image_proposal(img_path):
	img = skimage.io.imread(img_path)
	img_lbl, regions = selectivesearch.selective_search(img, scale=15, sigma=0.5, min_size=500)
	candidates = set()
	images = []
	vertices = []
	for r in regions:
		if r['rect'] in candidates:
			continue
		proposal_img, proposal_vertice = prep.clip_pic(img, r['rect'])
		if len(proposal_img) == 0:
			continue
		x, y, w, h = r['rect']
		if w == 0 or h == 0:
			continue
		[a, b, c] = np.shape(proposal_img)
		if a == 0 or b == 0 or c == 0:
			continue
		im = Image.fromarray(proposal_img)
		array_img = Image.fromarray(proposal_img) 
		resize_img = resize_image(array_img, 227, 227)
		img_float = pil_to_nparray(resize_img)
		images.append(img_float)
		vertices.append(r['rect'])
	return images, vertices
def predict():
	MODEL_SAVE_PATH = "fine_tune_for_rcnn/model/"
	num_classes= 95 #the classes of our data set butterfly
	refine_layers = []
	global_step = tf.Variable(0,trainable=False)
	#Input and output
	xs = tf.placeholder(tf.float32, [None, 227, 227, 3])
	keep_prob = tf.placeholder(tf.float32)
	#Initialize model
	model = AlexNet(xs, keep_prob, num_classes, refine_layers)
	# output of the AlexNet labels
	score = tf.nn.softmax(model.fc8)
	saver = tf.train.Saver()

	with tf.Session() as sess:
		tf.initialize_all_variables().run()
		#load the trained model
		if os.path.isfile(MODEL_SAVE_PATH+'checkpoint'):
			ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess,ckpt.model_checkpoint_path)
				global_step = int (ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
				predictfeatures=[]
				imgs = []
				verts = []
				img_path = 'demoimages'
				listings = os.listdir(img_path)
				for test_file in listings:
					#load the image and get the proposal region
					imgs, verts = image_proposal(img_path+'/'+test_file)
					#when the proposal region is too much, and the gpu memory is not enough, random select some
					'''
					imagespredict = []
					vertspredict = []
				
					for i in range(len(imgs)):
						#if i%3==0:
							imagespredict.append(imgs[i])
							vertspredict.append(verts[i])
					print(len(imagespredict))
					'''

					#get the output of the network
					predictfeatures = sess.run(score, feed_dict={xs:imgs,keep_prob:1.0})
					#find the predicted labels for all regions
					prediction = np.argmax(predictfeatures,1)

					#remove the 0 label(background)
					predictlist= prediction.tolist()
					while 0 in predictlist:
						predictlist.remove(0)
					c = (Counter(predictlist).most_common(1))
					maxlabel = c[0][0]
					results = []
					
					for i in range(len(predictfeatures)):
						if prediction[i] == maxlabel:
							results.append(verts[i])
					totalx = 0
					totaly = 0
					totalw = 0
					totalh = 0
					for x, y, w, h in results:
						totalx = totalx + x
						totaly = totaly + y
						totalw = totalw + w
						totalh = totalh + h
					x = totalx / len(results)
					y = totaly / len(results)
					w = totalw / len(results)
					h = totalh / len(results)
					img = skimage.io.imread(img_path+'/'+test_file)
					fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6)) 
					ax.imshow(img)
					plt.rcParams['font.sans-serif']=['SimHei']
					rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
					res = buttfly_classes.class_names[maxlabel-1]
					plt.text(200,200,res, bbox = dict(facecolor = "r"))
					plt.title(test_file)
					ax.add_patch(rect)
					plt.show()
			
				

		else:
			print("No model now!")
if __name__ == '__main__':
	predict()