import os
import pickle
import numpy as np
import tensorflow as tf
from datetime import datetime
from AlexNet import AlexNet
import preprocessing_rcnn as prep

MODEL_SAVE_PATH = "fine_tune_for_rcnn/model/"
MODEL_NAME = "fine_tune_rcnn_model.ckpt"
LEARING_RATE_BASE = 0.9
LEARNING_RATE_DECAY = 0.9
TRAINING_STEPS = 30000
KEEP_PROB = 0.5
#learning_rate = 0.01
batch_size = 128
trainfile_set = []
num_classes= 95 #the classes of our data set butterfly
refine_layers = ['fc8'] #we only refine the last layer, other layer is load from a trained one
train_layers = ['pool1','norm1','conv2','pool2','norm2','conv3','conv4','conv5','pool5','fc6','fc7','fc8']

global_step = tf.Variable(0,trainable=False)

#Input and output
x = tf.placeholder(tf.float32, [None, 227, 227, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

#Initialize model
model = AlexNet(x, keep_prob, num_classes, refine_layers)
# output of the AlexNet
score = model.fc8

var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
#loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score, labels = y))  
'''
#exponential learning rate decay
learning_rate = tf.train.exponential_decay(LEARING_RATE_BASE, global_step,100,LEARNING_RATE_DECAY)
#gradient descent 
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
with tf.control_dependencies([train_step]):
		train_op = tf.no_op(name = 'train')
		'''
learning_rate = tf.train.exponential_decay(LEARING_RATE_BASE, global_step,100,LEARNING_RATE_DECAY)
gradients = tf.gradients(loss, var_list)
gradients = list(zip(gradients, var_list))
 
# Create optimizer and apply gradient descent to the trainable variables
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.apply_gradients(grads_and_vars=gradients,global_step=global_step)

correct_prediction = tf.equal(tf.argmax(score,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

saver = tf.train.Saver()


with tf.Session() as sess:
	tf.initialize_all_variables().run()
	#load the trained model
	if os.path.isfile(MODEL_SAVE_PATH+'checkpoint'):
		ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess,ckpt.model_checkpoint_path)
			global_step = int (ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
			print("Continue training!")
	else:
		print("No model now!")
		model.load_initial_weights(sess)
	#load the validation data, the network should not dropout this time
	#get the validation data
	if os.path.isfile('dataset/valdataset.pkl'):
		valimages, vallabels = pickle.load(open('dataset/valdataset.pkl', 'rb'))
	else:
		print('No validation dataset!')
	print(np.array(valimages).shape,np.array(vallabels).shape)
	validate_feed = {x: valimages, y:vallabels, keep_prob:1.0}
	listings = os.listdir('dataset')
	for train_file in listings:
		if train_file == 'valdataset.pkl':
			continue
		trainfile_set.append(train_file)
	print(trainfile_set)
	traindataposition = 0 #keep the data file in trainfile_set
	batchposition = 0 #keep the position of the data in one file 
	keep_train = True #if the train set file should change, false means yes
	trainimages,trainlabels,unuseimagestest,unuselabelstest = prep.load_from_pkl('dataset/'+trainfile_set[traindataposition])
	dataset_size = len(trainimages)

	for i in range(TRAINING_STEPS):
		if i < global_step:
			continue
		else:
			#after 100 train step, get the validation results and then save the model
			if i % 100 == 0:
				validate_acc = sess.run(accuracy, feed_dict = validate_feed)
				print("After %d training step(s), validation accuracy " "using average model is %g " %(i, validate_acc))
				saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME), global_step=global_step)
			
			#set the datasize of the training data
			start = (batchposition*batch_size)%dataset_size
			batchposition = batchposition + 1
			end = start+batch_size
			if end > dataset_size:
				end = dataset_size
				keep_train = False
			#end = min(start+batch_size, dataset_size)
			#print("train step: %d"%i)
			#print(trainfile_set[traindataposition])
			sess.run(train_op, feed_dict={x:trainimages[start:end],y:trainlabels[start:end],keep_prob:KEEP_PROB})
			if not keep_train :
				traindataposition = (traindataposition+1)%len(trainfile_set) #keep the data file in trainfile_set
				trainimages,trainlabels,unuseimagestest,unuselabelstest = prep.load_from_pkl('dataset/'+trainfile_set[traindataposition])
				batchposition = 0
				keep_train = True
				dataset_size = len(trainimages)
		#test_acc = sess.run(accuracy, feed_dict= test_feed)
		#print("After %d training step(s), test accuracy using average " "model is %g"%(TRAINING_STEPS, test_acc))