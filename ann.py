from abc import ABC, abstractmethod

# To consider/add:
# 1. Helper constructor for multilayer perceptron, with given activation.

class ann(ABC):
	## ann - Artifical Neural Network
	# Parent class for various implementations		
	
	@abstractmethod
	def __str__(this):
		# Child classes should implement their display.
		pass	@abstractmethod	def forward(this,x):		# forward x through the network.		pass
		
class ann_by_layers(ann):
	
	## ann_by_layers - Artificial Neural Network constructed from a layer array.
	# The assumption here is the network is sequential, input data x flows from layers[n] to layers[n+1].
	
	def __init__(this, layers = [], name = ''):	
		this.name = name
		this.layers = layers		
		
	def __str__(this):
		return this.__class__.__name__ + " class with name " + this.name
					def forward(this,x):		# forward x through the layers.		y = x		for layer in this.layers:			y = layer.forward(y)		return y