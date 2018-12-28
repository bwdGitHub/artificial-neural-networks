import layers
import unittest
import numpy as np
import numpy.testing as nptest

# To add:
# 1. Test fc layer - construction with different inputs, forward with different inputs, backward with different inputs.

class test_fc_layer(unittest.TestCase):
	
	def __init__(this, *args, **kwargs):
		
		# Parameter for valid inputs to fc layer constructor. 
		this.validInputs = {
			'One' : {'NumHidden' : 1},
			'Many' : {'NumHidden' : 10}, 
			'WithWeights' : {'NumHidden' : 3, 'W' : np.ones((2,3))},
			'WithBias' : {'NumHidden' : 5, 'b' : np.ones((1,5))}
			}
			
		# Super Constructor
		super(test_fc_layer, this).__init__(*args,**kwargs)
	
	def assertEqualProperties(this, obj, *args, **kwargs):
		for key in kwargs:
			with this.subTest():
				# Hack to test np arrays separately to other kwargs.
				if not isinstance(kwargs[key], np.ndarray):
					this.assertEqual(getattr(obj, key), kwargs[key])
				else:
					nptest.assert_array_equal(getattr(obj, key), kwargs[key])					
	
	def test_constructor(this):
		
		# Test the constructor
		for input_key in this.validInputs:
			with this.subTest():
				layer = layers.fc(**this.validInputs[input_key])
				this.assertIsInstance(layer, layers.fc)
				this.assertEqualProperties(layer, **this.validInputs[input_key])
			
	def test_forward(this):
		
		# Test the forward method
		
		W = np.random.randint(low = -5, high = 5, size = (4, 5))
		b = np.random.randint(low = -5, high = 5, size = (4, 1))
		x = np.random.randint(low = -5, high = 5, size = (5, 1))
		layer = layers.fc(NumHidden = 4, W = W, b = b)
		y_act = layer.forward(x)
		y_exp = np.dot(W,x) + b
		
		# Assert with err_msg logs, since using random arrays.
		
		nptest.assert_array_equal(y_act,y_exp, err_msg = "fc layer forward incorrect for \n W= \n {} \n and \n b= \n {}".format(W,b))
		
	def test_W_gradient(this):
	
		# Test the gradients with respect to W
		W = np.random.randint(low = -5, high = 5, size = (4, 5))
		b = np.random.randint(low = -5, high = 5, size = (4, 1))
		x = np.random.randint(low = -5, high = 5, size = (5, 1))
		layer = layers.fc(NumHidden = 4, W = W, b = b)
		for i in range(4):
			for j in range(5):
				dydWij_act = layer.W_gradient(x,i,j)
				dydWij_exp = np.zeros((4,1))
				dydWij_exp[i] = x[j]
				nptest.assert_array_equal(dydWij_act, dydWij_exp, err_msg = "fc layer W_gradient incorrect for \n W= \n {} \n and \n b= \n {}".format(W,b))
				
	def test_b_gradient(this):
	
		# Test the gradients with respect to b
		W = np.random.randint(low = -5, high = 5, size = (4, 5))
		b = np.random.randint(low = -5, high = 5, size = (4, 1))
		x = np.random.randint(low = -5, high = 5, size = (5, 1))
		layer = layers.fc(NumHidden = 4, W = W, b = b)
		for i in range(4):
			dydb_act = layer.b_gradient(x,i)
			dydb_exp = np.zeros((4,1))
			dydb_exp[i] = 1
			nptest.assert_array_equal(dydb_act, dydb_exp, "fc layer b_gradient incorrect for \n W= \n {} \n and \n b= \n {}".format(W,b))
		
	def test_x_gradient(this):
	
		# Test the gradients with respect to x
		W = np.random.randint(low = -5, high = 5, size = (4, 5))
		b = np.random.randint(low = -5, high = 5, size = (4, 1))
		x = np.random.randint(low = -5, high = 5, size = (5, 1))
		layer = layers.fc(NumHidden = 4, W = W, b = b)
		for i in range(4):
			dydxi_act = layer.x_gradient(i)
			dydxi_exp = W[:,i]
			nptest.assert_array_equal(dydxi_act, dydxi_exp, "fc layer x_gradient incorrect for \n W= \n {} \n and \n b= \n {}".format(W,b))


		
if __name__ == '__main__':
	unittest.main()