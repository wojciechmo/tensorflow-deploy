import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import json
import sys

def preprocess_image(img):
	
	img = cv2.resize(img,(width,height))
	img = img[:,:,::-1] # BGR -> RGB
	img = img/255.0
	x_data = np.expand_dims(img,0)
	
	return x_data

def read_classes_names(filepath):
	
	# in imagenet_names.txt there are 1000 valid classes, while model predicts also 'zero' class
	classes = ['none']
	with open(filepath, 'r') as f:
		lines = f.read().splitlines()
		for line in lines:
			idx, name = line.split(':')
			idx = int(idx)
			name = name[2: -2]
			classes.append(name.lower())
			
	return classes

if __name__ == "__main__":

	module = hub.Module('https://tfhub.dev/google/imagenet/resnet_v2_50/classification/1', trainable=False)
	height, width = hub.get_expected_image_size(module)
	x = tf.placeholder(tf.float32, shape=(None, height, width, 3), name='x')
	logits = module(x) 
	probabilities = tf.nn.softmax(logits, axis=1, name='probabilities')
	_, predictions = tf.nn.top_k(logits, k=5, name='predictions')

	predictions_renamed = tf.identity(predictions, name='predictions_renamed')

	img = cv2.imread('tiger.jpg')
	x_data = preprocess_image(img)

	# init_vars = tf.global_variables_initializer()
	init_vars = tf.initialize_variables(tf.all_variables(), name='init_vars')

	sess=tf.Session()
	sess.run(init_vars)

	probs, top_k = sess.run([probabilities, predictions], feed_dict={x: x_data})
	top_k = top_k[0]
	top_k_probs = [probs[0][idx] for idx in top_k]
	classes = read_classes_names('imagenet_names.txt')
	top_k_names = [classes[idx] for idx in top_k]

	print 'Top 5 predicitions:'
	for idx, prob, name in zip(top_k, top_k_probs, top_k_names):
		print '--class:', name, '--probability:', prob'
		
	# for C/C++ deployment, it can also be used for Java but in such scenario one should use JavaCpp package to load model
	tf.train.write_graph(sess.graph_def, './model/', 'graph.pb', as_text=False)

	# for Java deployment
	# tf.saved_model.simple_save(sess, './saved_model', inputs={"x": x}, outputs={"predictions": predictions, "probabilities": probabilities})

	# for JavaScript deployment
	# 1. it's impossible to pass predicitons:1 instead of predicions:0 to tfjs converter, hence tf.identity
	# 2. no idea why these strings are passed as keys to inputs and outputs arguments, both tensorflowjs_converter as well as javascript itself can only reference operations by oryginal graph names
	# tf.saved_model.simple_save(sess, './saved_model', inputs={"x": x}, outputs={"predictions": predictions_renamed, "probabilities" : probabilities})
