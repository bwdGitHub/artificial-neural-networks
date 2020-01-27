from abc import ABC, abstractmethod
import numpy as np

# To Do/Consider:
# 1. Remove indexed return for gradients?
# 2. Fix down a gradient API?
# 3. Get trainable parameters API/Implementation? Default in ABC should be {}?
# 4. Adapt layers for multiple observations.

class layer(ABC):
	## layer - Abstract base class for layers.	
	
	@abstractmethod
	def forward(this, *args, **kwargs):
		pass		
		
	@abstractmethod
	def get_training_parameters(this, *args, **kwargs):
		pass

	@abstractmethod
	def initialize_training_parameters(this, *arg, **kwargs):
		pass

class fc(layer):
	
	## fc - A fully connected layer
	
	# Input is expected to be a vector x of size (n,1). Specify InputSize as n at layer construction.

	def __init__(this, InputSize, NumHidden, W=[], b = []):
		# Constructor - Initialize attributes.
		
		this.InputSize = InputSize
		this.NumHidden = NumHidden
		this.W = W
		this.b = b

	def __eq__(this, layer):
		return (isinstance(layer, fc) and
			np.array_equal(this.W, layer.W) and
			np.array_equal(this.b, layer.b) and
			this.InputSize == layer.InputSize and
			this.NumHidden == layer.NumHidden )

	def initialize_training_parameters( this, f = np.random.normal ):
		# Initialize the training parameters.
		this.W = f(size = (this.NumHidden, this.InputSize))
		this.b = f(size = (this.NumHidden, 1))	
		
	def forward(this, x):
		# Forward x.
		return np.dot(this.W, x) + this.b	

	def get_training_parameters(this, x = []):
		# Return dictionary of trainable parameters, optionally with gradients.
		if len(x)==0:
			params = { 
			'Weights': {'Name': 'W'},
			'Bias': {'Name':'b'}
			}
		else:
			params = { 
				'Weights' : {'Name': 'W', 'Gradient': this.W_gradient(x)},
				'Bias' : {'Name':'b', 'Gradient': this.b_gradient(x)}
					}
		return params
		
	def W_gradient(this, x, i=[], j=[]):
		# Gradient of y = this.forward(x) w.r.t. W[i,j]	
		
		# If i==[], j==[], tensor return
		if i==[] and j==[]:
			dydW = np.zeros((this.NumHidden, this.NumHidden, x.shape[0]))
			for ii in range(this.NumHidden):
				for jj in range(x.shape[0]):
					dydW[ii,ii,jj] = x[jj]
			return dydW
		
		# dy/dW[i,j]
		
		dydWij = np.zeros((this.NumHidden,1))
		dydWij[i] = x[j]
		return dydWij
		
	def b_gradient(this, x, i=[]):
		# Gradient of y = this.forward(x) w.r.t. b[i]
		
		# If i == [], return tensor
		if i == []:
			dydb = np.eye(this.NumHidden)
			return dydb
		
		# dy/db[i]
		dydb = np.zeros((this.NumHidden,1))
		dydb[i] = 1
		return dydb
		
	def x_gradient(this, x, i=[]):	
		# Gradient of y = this.forward(x) w.r.t. x[i]
		
		# If i==[] return tensor
		if i == []:
			dydx = this.W
			return dydx
		
		# dy/dx[i]
		
		return this.W[:,i]

class relu(layer):
	# ReLu layer - rectified linear unit, given a tensor x, replace each element of xi of x by max(xi,0).

	def __eq__(this, layer):
		return isinstance(layer, relu)

	def forward(this,x):
		return np.maximum(x,0)

	def get_training_parameters(this,x=[]):
		return {}

	def initialize_training_parameters(this, f):
		pass

	def x_gradient(this, x):
		# Gradient of y = this.forward(x) w.r.t. x
		dydx = np.eye(x.shape[0])
		idx = x[:,0]<0
		dydx[idx, idx] = 0
		return dydx
		
class mse(layer):
	# Mean square error loss layer

	def __eq__(this, layer):
		return isinstance(layer, mse)

	def get_training_parameters(this, x=[]):
		return {}

	def initialize_training_parameters(this, f):
		pass

	def forward(this, x, y):
		# Compute mean square error between x and y
		return np.sum(np.power(x-y,2))

	def x_gradient(this, x, y):
		# Compute the gradient of mean square error
		return 2*(x-y)