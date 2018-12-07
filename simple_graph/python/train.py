import tensorflow as tf
import numpy as np

if __name__ == "__main__":

	x = tf.placeholder(tf.float32, shape=(4), name='x')
	w = tf.Variable(tf.constant([1.0, 2.0, 3.0, 4.0]), name='w')
	y = tf.multiply(x, w, name='y')

	# init_vars = tf.global_variables_initializer()
	init_vars = tf.initialize_variables(tf.all_variables(), name='init_vars')
	
	saver = tf.train.Saver()
	sess=tf.Session()
	
	sess.run(init_vars)
	x_data = np.array([2.0, 2.0, 2.0, 2.0])
	y_data = sess.run(y, feed_dict={x: x_data})
	print "Input vector:"
	print x_data
	print "Output vector:"
	print y_data

	# for Java and Javascript deployment
	tf.saved_model.simple_save(sess, './saved_model', inputs={"x": x}, outputs={"y": y})

	# for C++ and C deployment, it can also be used for Java with JavaCpp package to load model
	tf.train.write_graph(sess.graph_def, './model/', 'graph.pb', as_text=False)

	# saver.save(sess, "saved_model/model.ckpt")
