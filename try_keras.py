from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np

def make_model():
	#Get back the convolutional part of a VGG network trained on ImageNet
	model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

	#Create your own input format (here 3x200x200)
	input = Input(shape=(160,320, 3),name = 'image_input')

	#Use the generated model 
	output_vgg16_conv = model_vgg16_conv(input)

	# freeze vgg16 conv layers
	for layer in model_vgg16_conv.layers:
	    layer.trainable = False

	#model_vgg16_conv.summary()

	#Add the fully-connected layers 
	x = Flatten(name='flatten')(output_vgg16_conv)
	x = Dense(4096, activation='relu', name='fc1')(x)
	x = Dense(4096, activation='relu', name='fc2')(x)
	x = Dense(1, activation='tanh', name='predicton_steering')(x)

	#Create your own model 
	return Model(input=input, output=x)

#In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
my_model = make_model()
#my_model.summary()
my_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
history = my_model.fit(X, y, batch_size=3, np_epoch=10)
