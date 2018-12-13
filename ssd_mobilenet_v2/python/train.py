import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import sys

def preprocess_input(img):
	
	img = img[:,:,::-1] # BGR -> RGB
	img = img/255.0
	x_data = np.expand_dims(img,0)
	
	return x_data

def show_output(img, boxes, scores, names):

	
	colors = {name:np.random.uniform(0, 256, size=(3)).astype(np.int32)  for name in list(set(names))}
	
	height, width, _ = img.shape 
	
	for box, score, name in zip(boxes, scores, names):
			
		y1 = int(box[0] * height)
		x1 = int(box[1] * width)
		y2 = int(box[2] * height)
		x2 = int(box[3] * width)
		
		if score > 0.2:
			
			color =  tuple(map(int, colors[name]))
			cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(img, name, (x1, y1), font, 1, color ,2, cv2.LINE_AA)
			
			print '--name:', name, '--score:', score, '--box:', [x1, y1, x2, y2]

	cv2.imshow('image', img)
	cv2.moveWindow('image', 0, 0)
	cv2.waitKey()

if __name__ == '__main__':
	
	module = hub.Module("https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1")
	
	# tf.get_default_graph().as_graph_def().node[-1].name = 'new_name'
	# print tf.get_default_graph().as_graph_def().node[-1].name
	
	#x = tf.placeholder(tf.float32, [1, None, None, 3], name='module_apply_default/image_tensor')
	x = tf.get_default_graph().get_tensor_by_name("module/image_tensor:0")
	
	module_output = module(x, as_dict=True)
	boxes = module_output["detection_boxes"]
	scores = module_output["detection_scores"]
	names = module_output["detection_class_entities"]
		
	# print boxes -> Tensor("module_apply_default/hub_input/strided_slice:0", shape=(?, 4), dtype=float32)
	# print scores -> Tensor("module_apply_default/hub_input/strided_slice_1:0", shape=(?, 1), dtype=float32)
	# print names -> Tensor("module_apply_default/hub_input/index_to_string_1_Lookup:0", shape=(?, 1), dtype=string)

	# init_vars = tf.global_variables_initializer()
	init_vars = tf.initialize_variables(tf.all_variables(), name='init_vars')
	init_tables = tf.initialize_all_tables(name='init_all_tables')

	sess = tf.Session()
	sess.run(init_vars)		
	sess.run(init_tables)
				
	img = cv2.imread('./bicycles.png')
	# img = cv2.resize(img,(300,300))
	x_data = preprocess_input(img)
	
	boxes_data, scores_data, names_data = sess.run([boxes, scores, names], feed_dict={x: x_data})
	
	show_output(img, boxes_data, scores_data, names_data)
