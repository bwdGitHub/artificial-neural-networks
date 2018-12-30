from abc import ABC, abstractmethod
import numpy as np

# To Do/Consider:
# 1. Remove indexed return for gradients?# 2. Fix down a gradient API?# 3. Get trainable parameters API/Implementation?

class layer(ABC):
	## layer - Abstract base class for layers.	
	
	@abstractmethod
	def forward(this, *args, **kwargs):
		pass
		
	@abstractmethod
	def backward(this, *args, **kwargs):
		pass
		
	@abstractmethod	def get_training_parameters(this, *args, **kwargs):		pass
class fc(layer):
	
	## fc - A fully connected layer

	def __init__(this, NumHidden, W=[], b = []):
		# Constructor - Initialize attributes.
		
		this.NumHidden = NumHidden
		this.W = W
		this.b = b
		
	def forward(this, x):
		# Forward x.
		return np.dot(this.W, x) + this.b
	
	def backward(this):
		print("Backward for fc not implemented")
		raise NotImplementedError	def get_training_parameters(this, x = []):		# Return dictionary of trainable parameters, optionally with gradients.		if x == []:			params = { 
			'Weights': {'Name': 'W'},
			'Bias': {'Name':'b'}
			}		else:			params = { 				'Weights' : {'Name': 'W', 'Gradient': this.W_gradient(x)},				'Bias' : {'Name':'b', 'Gradient': this.b_gradient(x)}					}		return params
		
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
		
		return this.W[:,i]class relu(layer):	def forward(this,x):		return np.maximum(x,0)	def backward(this):		return NotImplementedError	def get_training_parameters(this,x=[]):		return {}	def x_gradient(this, x):		# Gradient of y = this.forward(x) w.r.t. x		dydx = np.eye(x.shape[0])		idx = x[:,0]<0		dydx[idx, idx] = 0		return dydx		