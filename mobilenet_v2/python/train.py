import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

class Model(tf.keras.Model):
	
	def __init__(self):
		super(Model, self).__init__()
		self.layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4")
		self.softmax = tf.keras.layers.Softmax()

	def call(self, x):
		x = self.layer(x)
		x = self.softmax(x)
		values, indexes = tf.math.top_k(x, 5)

		return indexes

def preprocess_input(img):
	
	img = img[:,:,::-1] # BGR -> RGB
	img = np.expand_dims(img/255.0, 0).astype(np.float32)
	
	return img

if __name__ == '__main__':

	imagenet_labels = np.array(open('/tmp/assets/ImageNetLabels.txt').read().splitlines())
		
	model = Model()
	model.build((1, 224, 224, 3))
	print (model.summary())

	img = cv2.imread('/tmp/assets/lion.jpg')
	img = cv2.resize(img, (224, 224))
	input_data = preprocess_input(img)

	print ('test model performance')
	output_data = model(input_data)
	for idx in output_data[0]:
		print (imagenet_labels[idx])

	run_model = tf.function(lambda x : model(x))
	mockup_input_data = tf.constant(1.0, shape=[1, 224, 224, 3])
	concrete_func = run_model.get_concrete_function(mockup_input_data)
	
	converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
	tflite_model = converter.convert()
	with open("/tmp/assets/model.tflite", "wb") as file:
		file.write(tflite_model)
	
	interpreter = tf.lite.Interpreter(model_path="/tmp/assets/model.tflite")
	interpreter.allocate_tensors()

	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	print ('verify tflite model performance')	
	interpreter.set_tensor(input_details[0]['index'], input_data)
	interpreter.invoke()
	output_data = interpreter.get_tensor(output_details[0]['index'])
	for idx in output_data[0]:
		print (imagenet_labels[idx])
		


