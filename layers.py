from abc import ABC, abstractmethod
import numpy as np

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

	def __init__(this, input_size, output_size, W, b):
		this.W = W
		this.b = b
		
	def forward(this, x):
		return np.dot(this.W, x) + this.b
	
	def backward(this):
		print("Backward for fc not implemented")
		raise NotImplementedError