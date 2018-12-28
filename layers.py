from abc import ABC, abstractmethod
import numpy as np

# To Do/Consider:
# 1. fc layer gradients format could be more general, but would need a 3d-array for dy/dW

class layer(ABC):
	## layer - Abstract base class for layers.	
	
	@abstractmethod
	def forward(this, *args, **kwargs):
		pass
		
	@abstractmethod
	def backward(this, *args, **kwargs):
		pass
		

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
		raise NotImplementedError
		
	def W_gradient(this, x, i, j):
		# Gradient of y = this.forward(x) w.r.t. W[i,j]	
		# dy/dW[i,j]
		dydWij = np.zeros((this.NumHidden,1))
		dydWij[i] = x[j]
		return dydWij
		
	def b_gradient(this, x, i):
		# Gradient of y = this.forward(x) w.r.t. b[i]
		# dy/db[i]
		dydb = np.zeros((this.NumHidden,1))
		dydb[i] = 1
		return dydb
		
	def x_gradient(this, i=[]):	
		# Gradient of y = this.forward(x) w.r.t. x[i]
		# dy/dx[i]
		
		return this.W[:,i]