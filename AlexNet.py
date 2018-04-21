import tensorflow as tf
import numpy as np

class AlexNet(object):
	def __init__(self, x, keep_prob, num_classes, skip_layer,weights_path = 'DEFAULT'):
		#x is the input images
		#skip_layer are the layers that will be trained again
		#num_classes is the output classes after finetuing
		self.X = x
		self.NUM_CLASSES = num_classes
		self.KEEP_PROB = keep_prob
		self.SKIP_LAYER = skip_layer
		if weights_path == 'DEFAULT':    
			#bvlc_alexnet.npy is a collection of weights of the trained AlexNet get from toranto uni
			self.ALEXNET_WEIGHTS_PATH = 'bvlc_alexnet.npy'
		else:
			self.ALEXNET_WEIGHTS_PATH = weights_path
		# Call the create function to build the computational graph of AlexNet
		self.create()
	def create(self):
		#conv1
		conv1 = conv(self.X, 11, 11, 96, 4, 4, padding = 'VALID', name = 'conv1')
		
		#lrn1 and pool1 
		#lrn1 = tf.nn.lrn(conv1,4,bias=1.0,alpha = 0.001/9,beta = 0.75, name = 'norm1')
		pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1],padding ='VALID', name='pool1')
		lrn1 = tf.nn.lrn(pool1,bias=1.0,depth_radius=2 ,alpha = 2e-05,beta = 0.75, name = 'norm1')
		#conv2
		conv2 = conv(lrn1, 5, 5, 256, 1, 1, groups = 2, name = 'conv2')
		
		#lrn2 and pool2
		#lrn2 = tf.nn.lrn(conv2,4,bias=1.0,alpha = 0.001/9,beta = 0.75, name = 'norm2')
		pool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides=[1,2,2,1],padding ='VALID', name='pool2')
		lrn2 = tf.nn.lrn(pool2,bias = 1.0,depth_radius=2 ,alpha = 2e-05,beta = 0.75, name = 'norm2')
		#conv3
		conv3 = conv(lrn2, 3, 3, 384, 1, 1, name = 'conv3')
		#conv4
		conv4 = conv(conv3, 3, 3, 384, 1, 1, groups = 2, name = 'conv4')
		
		#conv5
		conv5 = conv(conv4, 3, 3, 256, 1, 1, groups = 2, name = 'conv5')
		
		pool5 = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides=[1,2,2,1],padding ='VALID', name='pool5')
		#fcIn
		fcIn = tf.reshape(pool5, [-1, 256 * 6 * 6])  
		with tf.variable_scope('fc6') as scope:
			#weights = tf.Variable(tf.truncated_normal([256 * 6 * 6,4096],dtype= tf.float32,stddev= 1e-1),name = 'weights')
			weights = tf.get_variable('weights',shape=[256 * 6 * 6,4096],trainable=True)
			#biases = tf.Variable(tf.constant(0.0, shape=[4096],dtype = tf.float32),trainable= True, name = 'biases')
			biases = tf.get_variable('biases', shape = [4096],trainable=True)
			self.fc6 = tf.nn.xw_plus_b(fcIn,weights,biases,name = scope.name)
			fc6 = tf.nn.relu(self.fc6)
		#set the dropout keepPro 0.5
		dropout1 =  tf.nn.dropout(fc6,self.KEEP_PROB)
		#fc2
		with tf.variable_scope('fc7') as scope:
			#weights = tf.Variable(tf.truncated_normal([4096,4096],dtype= tf.float32,stddev= 1e-1),name = 'weights')
			weights = tf.get_variable('weights',shape=[4096,4096],trainable=True)
			#biases = tf.Variable(tf.constant(0.0, shape=[4096],dtype = tf.float32),trainable= True, name = 'biases')
			biases = tf.get_variable('biases', shape = [4096],trainable=True)
			self.fc7 = tf.nn.xw_plus_b(dropout1,weights,biases,name = scope.name)
			fc7 = tf.nn.relu(self.fc7)
		dropout2 =  tf.nn.dropout(fc7,self.KEEP_PROB)
		#fc3
		with tf.variable_scope('fc8') as scope:
			#weights = tf.Variable(tf.truncated_normal([4096,self.NUM_CLASSES],dtype= tf.float32,stddev= 1e-1),name = 'weights')
			weights = tf.get_variable('weights',shape=[4096,self.NUM_CLASSES],trainable=True)
			#biases = tf.Variable(tf.constant(0.0, shape=[self.NUM_CLASSES],dtype = tf.float32),trainable= True, name = 'biases')
			biases = tf.get_variable('biases', shape = [self.NUM_CLASSES],trainable=True)
			self.fc8 = tf.nn.xw_plus_b(dropout2,weights,biases,name = scope.name)
	def load_initial_weights(self, session):
		#load the weights from the file.
		weights_dict = np.load(self.ALEXNET_WEIGHTS_PATH, encoding = 'bytes').item()
		# Loop over all layer names stored in the weights dict
		for op_name in weights_dict:
			# Check if the layer is one of the layers that should be trained
			if op_name not in self.SKIP_LAYER:
				with tf.variable_scope(op_name, reuse = True):
					# Loop over list of weights/biases and assign them to their corresponding tf variable
					for data in weights_dict[op_name]:
						#only one is the biases
						if len(data.shape) == 1:
							var = tf.get_variable('biases', trainable = True)
							session.run(var.assign(data))
						# Weights
						else:
							var = tf.get_variable('weights', trainable = True)
							session.run(var.assign(data))
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):
	input_channels = int(x.get_shape()[-1])
	#convolution
	convolve = lambda i, k: tf.nn.conv2d(i, k, strides = [1, stride_y, stride_x, 1],padding = padding)
	with tf.variable_scope(name) as scope:
		# Create tf variables for the weights and biases of the conv layer
		weights = tf.get_variable('weights', shape = [filter_height, filter_width, input_channels/groups, num_filters],trainable= True)
		biases = tf.get_variable('biases', shape = [num_filters],trainable= True)  
		if groups == 1:
			conv = convolve(x, weights)
		# In the cases of multiple groups, split inputs & weights and
		else:
			# Split input and weights and convolve them separately
			input_groups = tf.split(axis = 3, num_or_size_splits=groups, value=x)
			weight_groups = tf.split(axis = 3, num_or_size_splits=groups, value=weights)
			output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]
			# Concat the convolved output together again
			conv = tf.concat(axis = 3, values = output_groups)
		# Add biases 
		#bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
		bias = tf.nn.bias_add(conv,biases)
		# Apply relu function
		relu = tf.nn.relu(bias, name = scope.name)
		return relu