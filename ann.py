from abc import ABC, abstractmethodimport numpy as npimport layers

# To consider/add:
# 1. Helper constructor for multilayer perceptron, with given activation.

class ann(ABC):
	## ann - Artifical Neural Network
	# Parent class for various implementations		
	
	@abstractmethod
	def __str__(this):
		# Child classes should implement their display.
		pass	@abstractmethod	def forward(this,x):		# forward x through the network.		pass	@abstractmethod	def initialize_training_parameters(this, f = np.random.normal ):		# Initialize values of training parameters.		pass	@abstractmethod	def get_training_parameters(this, x = []):		# Get training parameters of the network		pass
		
class ann_by_layers(ann):
	
	## ann_by_layers - Artificial Neural Network constructed from a layer array.
	# The assumption here is the network is sequential, input data x flows from layers[n] to layers[n+1].
	
	def __init__(this, layers = [], name = ''):	
		this.name = name
		this.layers = layers		
		
	def __str__(this):
		return this.__class__.__name__ + " class with name " + this.name	def __eq__(this, net):		return (isinstance(net, ann_by_layers) and			this.layers == net.layers and			this.name == net.name)			
					def initialize_training_parameters(this, f = np.random.normal ):		# Initialize training parameters of the layers.		for layer in this.layers:			layer.initialize_training_parameters( f )	def get_training_parameters(this, x = []):		# Get training parameters of each layer		params = []		forwardX = (len(x)==0)		xhat = x		for layer in this.layers:			params.append(layer.get_training_parameters(x))			if forwardX:
				xhat = layer.forward(x)		return params	def forward(this,x):		# forward x through the layers.		y = x		for layer in this.layers:			y = layer.forward(y)		return yclass mlp(ann_by_layers):	## mlp - A convenience network constructor for creating multi-layer perceptrons.	# Given an input size, a sequence of hidden sizes, and an activation function a network of	# fully connected layers with those hidden sizes, joined via the activation function, is constructed.	def __init__(this, InputSize, HiddenSizes, Activation = layers.relu, name = ''):		l = []		numIter = len(HiddenSizes)		# Put InputSize at the end of HiddenSizes, can be accessed via -1 index.		HiddenSizes.append(InputSize)		for i in range(numIter):			fcLayer = layers.fc(InputSize = HiddenSizes[i-1], NumHidden = HiddenSizes[i])			activationLayer = Activation()			l.append(fcLayer)				l.append(activationLayer)		super(mlp, this).__init__(layers = l, name = name)	def cast_to_ann_by_layers(this):		return ann_by_layers( layers = this.layers )