import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

def read_classes_names(filepath):

	classes = []
	with open(filepath, 'r') as f:
		lines = f.read().splitlines()
		for line in lines:
			idx, name = line.split(':')
			idx = int(idx)
			name = name[2: -2]
			classes.append(name.lower())
			
	return classes

if __name__ == '__main__':

	module = hub.Module('https://tfhub.dev/deepmind/biggan-512/2', trainable=False)

	batch_size = 2
	truncation = 0.5
	z_latent = truncation * tf.random.truncated_normal([batch_size, 128])  # noise sample
	indexes = tf.random.uniform([batch_size], maxval=1000, dtype=tf.int32)  # classes indexes
	indexes_onehot = tf.one_hot(indexes, 1000)
	samples = module(dict(y=indexes_onehot, z=z_latent, truncation=truncation)) # shape [2, 512, 512, 3]

	#print indexes -> Tensor("random_uniform:0", shape=(2,), dtype=int32)
	#print samples -> Tensor("module_apply_default/G_trunc_output:0", shape=(2, 512, 512, 3), dtype=float32)

	# init_vars = tf.global_variables_initializer()
	init_vars = tf.initialize_variables(tf.all_variables(), name='init_vars')

	sess = tf.Session()
	sess.run(init_vars)

	samples_data, indexes_data = sess.run([samples, indexes])

	classes = read_classes_names('imagenet_names.txt')
	
	print 'Images generated:'
	for sample_data, idx in zip(samples_data, indexes_data):
		
		print idx, '->', classes[idx]
		sample_data = sample_data/2.0 + 0.5 # generated images are in range [-1, 1].
		sample_data = sample_data[:,:,::-1] # RGB -> BGR
		cv2.imwrite('imgs/' + classes[idx] + '.png', (sample_data * 255.0).astype(np.uint8))

	# for C/C++ deployment, JavaCpp package should be used for Java deployment
	tf.train.write_graph(sess.graph_def, './model/', 'graph.pb', as_text=False)
