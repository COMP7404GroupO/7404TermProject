import os
import pickle
import numpy as np
import tensorflow as tf
from datetime import datetime
from AlexNet import AlexNet
import preprocessing_rcnn as prep
from sklearn import svm

MODEL_SAVE_PATH = "fine_tune_for_rcnn/model/"
MODEL_NAME = "fine_tune_rcnn_model.ckpt"
num_classes= 95 #the classes of our data set butterfly
refine_layers = []
global_step = tf.Variable(0,trainable=False)

svm_training_dataset_file = 'datasetsvm/'

#Input and output
x = tf.placeholder(tf.float32, [None, 227, 227, 3])
keep_prob = tf.placeholder(tf.float32)

#Initialize model
model = AlexNet(x, keep_prob, num_classes, refine_layers)
# output of the AlexNet as the the input of the svm training
score = model.fc7


saver = tf.train.Saver()


with tf.Session() as sess:
	tf.initialize_all_variables().run()
	#load the trained model
	if os.path.isfile(MODEL_SAVE_PATH+'checkpoint'):
		ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess,ckpt.model_checkpoint_path)
			global_step = int (ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
			print("Start svm training!")
			'''
			for i in range(num_classes):
				print("Start the %dth svm training！" %(i+1))
				images, imagelabels, imagestest, imagelabelstest = prep.load_from_pkl(svm_training_dataset_file+str(i+1)+'svmdataset.pkl')
				print(svm_training_dataset_file+str(i+1)+'svmdataset.pkl')
				print(len(images))
				print(len(imagelabels))
				print(len(imagestest))
				print(len(imagelabelstest))
				features = sess.run(score, feed_dict={x:images,keep_prob:1.0}) #input as the svm training, imagelabels are the labels of the svm training labels
				testfeatures = sess.run(score, feed_dict={x: imagestest,keep_prob:0.1})
				print(np.array(features).shape)
				print(np.array(testfeatures).shape)
			'''
			i=27
			images, imagelabels, imagestest, imagelabelstest = prep.load_from_pkl(svm_training_dataset_file+str(i)+'svmdataset.pkl')
			print(svm_training_dataset_file+str(i)+'svmdataset.pkl')
			print(len(images))
			print(len(imagelabels))
			print(len(imagestest))
			print(len(imagelabelstest))
			print(imagelabelstest)
			features = sess.run(score, feed_dict={x:images,keep_prob:1.0}) #input as the svm training, imagelabels are the labels of the svm training labels
			testfeatures = sess.run(score, feed_dict={x: imagestest,keep_prob:1.0})
			svmmodel=svm.LinearSVC()
			svmmodel.fit(features,imagelabels)
			score = svmmodel.score(testfeatures,imagelabelstest)
			predictfeaturesone=[]
			for i in range(len(testfeatures)):
					predictfeaturesone.append(testfeatures[i])
					pred = svmmodel.predict(predictfeaturesone)
					predictfeaturesone.clear()
					print(pred)
			print(score)
			pickle.dump(svmmodel, open('svmmodel', 'wb'))
	else:
		print("No model now!")