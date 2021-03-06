from abc import ABC, abstractmethod
import numpy as np
import layers

# To consider/add:
# 2. The backprop_updates method is a mess.
# 3. Training parameters API/class.
# 4. Data handler/dispenser.
# 5. Mini-batch training.
# 6. Consider update_training_parameters - it assumes the update is to be taken from the current value.
# 		That is tied to gradient descent. Really this should be done via some sort of optimizer that computes the full replacement value.

class ann(ABC):
	## ann - Artifical Neural Network
	# Parent class for various implementations		
	
	@abstractmethod
	def __str__(this):
		# Child classes should implement their display.
		pass

	@abstractmethod
	def forward(this,x):
		# forward x through the network.
		pass

	@abstractmethod
	def initialize_training_parameters(this, f = np.random.normal ):
		# Initialize values of training parameters.
		pass

	@abstractmethod
	def get_training_parameters(this, x = []):
		# Get training parameters of the network
		pass
		
class ann_by_layers(ann):
	
	## ann_by_layers - Artificial Neural Network constructed from a layer array.
	# The assumption here is the network is sequential, input data x flows from layers[n] to layers[n+1].
	
	def __init__(this, layers = [], name = ''):	
		this.name = name
		this.layers = layers		
		
	def __str__(this):
		return this.__class__.__name__ + " class with name " + this.name

	def __eq__(this, net):
		return (isinstance(net, ann_by_layers) and
			this.layers == net.layers and
			this.name == net.name)
			
				
	def initialize_training_parameters(this, f = np.random.normal ):
		# Initialize training parameters of the layers.
		for layer in this.layers:
			layer.initialize_training_parameters( f )

	def get_training_parameters(this, x = []):
		# Get training parameters of each layer
		params = []
		forwardX = (len(x)!=0)
		xhat = x
		for layer in this.layers:
			params.append(layer.get_training_parameters(xhat))
			if forwardX:
				xhat = layer.forward(xhat)
		return params

	def forward(this,x):
		# forward x through the layers.
		return this.forward_to_layer(x, len(this.layers))
	
	def forward_to_layer(this, x, j):
		# forward x from layer 1 to layer j.
		y = x
		for i in range(j):
			y = this.layers[i].forward(y)
		return y

	def collect_forwards(this, x):
		# helper to grab the forwards of x at each layer once.		
		y = x
		forwards = [y]
		for i in range(len(this.layers)):
			y = this.layers[i].forward(y)
			forwards.append(y)
		return forwards
		
	def collect_mid_layer_gradients(this, x):
		# helper to grab the mid-layer gradients.
		forwards = this.collect_forwards(x)
		# Need an outputSize. Used the size of the final forward. - Will probably fail for >1d inputs.
		outputSize = forwards[-1].shape[0]		
		dxMidLayer = np.eye(outputSize)		
		dxMidLayerCollection = [dxMidLayer]
		
		for i in range(len(this.layers)-1, -1, -1):
			# Iterate backward through the layers			
			dxLayerI = this.layers[i].x_gradient(forwards[i])
			dxMidLayer = np.matmul(dxMidLayer, dxLayerI)
			dxMidLayerCollection.insert(0,dxMidLayer)
		return dxMidLayerCollection
			
	def backprop_updates(this, x, y, lossLayer, lr = 0.001):
		dxMidLayerCollection = this.collect_mid_layer_gradients(x)
		params = this.get_training_parameters(x)
		
		# precompute the network output for later use
		z = this.forward(x)
		
		# Initialize updates as empty list
		updates =[]
		
		# Iterate through the layers
		for i, layer_params in zip(range(len(this.layers)), params):
			layer_updates = {}
			# Iterate through the training parameters for layer i
			for param in layer_params:		
				
				# layer gradient w.r.t. parameter
				dydp = layer_params[param]['Gradient']
				
				# Mid-layer gradient product via chain rule
				# Current convention is tensordor over (1,0) axes
				# Because mid layer derivative is N1 x N2, but parameter derivative could be a tensor
				# e.g. for a fullyConnected, dydW is dim(y) x dim(W,1) x dim(W,2).
				# You want to sum over the output dimension, dim(y), which is currently in the 0 index.
				# The mid layer derivative sums over the last dimension, which is currently 1 because x is assumed to be a vector.
				# Also note - need to use i+1 in dxMidLayerCollection as it has a full backward x-gradient as the first entry.
				dMidLayers_dp = np.tensordot(dxMidLayerCollection[i+1], dydp, axes=(1,0))
				
				# Final loss gradient
				dLdx = lossLayer.x_gradient(z, y)
				dLdp = np.tensordot(dLdx, dMidLayers_dp, axes=(0,0))
				
				# Squeeze out first dimension - corresponds to the dimension of the loss output - assumed 1 for now.
				dLdp = np.squeeze(dLdp, axis = 0)
				# Re-add last singleton dimension - seems to get lost somehow
				if len(dLdp.shape)==1:
					dLdp = np.expand_dims(dLdp, -1)
				
				# Initialize layer_updates for this parameter, then insert updates using learn rate lr.
				layer_updates[param] = {}
				layer_updates[param]['Update'] = lr*dLdp
				layer_updates[param]['Name'] = layer_params[param]['Name']
			
			# Append the updates for a single layer to the updates list
			updates.append(layer_updates)
		return updates
		
	def update_training_parameters(this, updates):
		# Method to update training parameters
		# Iterate through layers
		for layer, update in zip(this.layers, updates):
			# Iterate through layer parameters
			for param in update:
				thisParamName = update[param]['Name']
				current = getattr(layer, thisParamName)
				new = current - update[param]['Update']
				setattr(layer, thisParamName, new)
		
class mlp(ann_by_layers):

	## mlp - A convenience network constructor for creating multi-layer perceptrons.
	# Given an input size, a sequence of hidden sizes, and an activation function a network of
	# fully connected layers with those hidden sizes, joined via the activation function, is constructed.

	def __init__(this, InputSize, HiddenSizes, Activation = layers.relu, name = ''):
		l = []
		numIter = len(HiddenSizes)
		# Put InputSize at the end of HiddenSizes, can be accessed via -1 index.
		HiddenSizes.append(InputSize)
		for i in range(numIter):
			fcLayer = layers.fc(InputSize = HiddenSizes[i-1], NumHidden = HiddenSizes[i])
			activationLayer = Activation()
			l.append(fcLayer)	
			l.append(activationLayer)
		super(mlp, this).__init__(layers = l, name = name)

	def cast_to_ann_by_layers(this):
		return ann_by_layers( layers = this.layers )
