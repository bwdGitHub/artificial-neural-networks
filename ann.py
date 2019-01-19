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
					def initialize_training_parameters(this, f = np.random.normal ):		# Initialize training parameters of the layers.		for layer in this.layers:			layer.initialize_training_parameters( f )	def get_training_parameters(this, x = []):		# Get training parameters of each layer		params = []		forwardX = (len(x)!=0)		xhat = x		for layer in this.layers:			params.append(layer.get_training_parameters(xhat))			if forwardX:
				xhat = layer.forward(xhat)		return params	def forward(this,x):		# forward x through the layers.		return this.forward_to_layer(x, len(this.layers))		def forward_to_layer(this, x, j):		# forward x from layer 1 to layer j.		y = x		for i in range(j):			y = this.layers[i].forward(y)		return y	def collect_forwards(this, x):		# helper to grab the forwards of x at each layer once.				y = x		forwards = [y]		for i in range(len(this.layers)):			y = this.layers[i].forward(y)			forwards.append(y)		return forwards			def collect_mid_layer_gradients(this, x):		# helper to grab the mid-layer gradients.		forwards = this.collect_forwards(x)		# Need an outputSize. Used the size of the final forward. - Will probably fail for >1d inputs.		outputSize = forwards[-1].shape[0]				dxMidLayer = np.eye(outputSize)				dxMidLayerCollection = [dxMidLayer]				for i in range(len(this.layers)-1, -1, -1):			# Iterate backward through the layers						dxLayerI = this.layers[i].x_gradient(forwards[i])			dxMidLayer = np.matmul(dxMidLayer, dxLayerI)			dxMidLayerCollection.insert(0,dxMidLayer)		return dxMidLayerCollection											class mlp(ann_by_layers):	## mlp - A convenience network constructor for creating multi-layer perceptrons.	# Given an input size, a sequence of hidden sizes, and an activation function a network of	# fully connected layers with those hidden sizes, joined via the activation function, is constructed.	def __init__(this, InputSize, HiddenSizes, Activation = layers.relu, name = ''):		l = []		numIter = len(HiddenSizes)		# Put InputSize at the end of HiddenSizes, can be accessed via -1 index.		HiddenSizes.append(InputSize)		for i in range(numIter):			fcLayer = layers.fc(InputSize = HiddenSizes[i-1], NumHidden = HiddenSizes[i])			activationLayer = Activation()			l.append(fcLayer)				l.append(activationLayer)		super(mlp, this).__init__(layers = l, name = name)	def cast_to_ann_by_layers(this):		return ann_by_layers( layers = this.layers )