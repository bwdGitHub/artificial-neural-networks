import unittest
from ann import ann, ann_by_layers
import io
import contextlibimport layersimport numpy.testing as nptestimport numpy as np

# To Consider/Add:
# 1. Mock test the base class
# 2. Test ann_by_layers with layer array inputs - maybe mock layers once/if input validation added.# 3. Test ann_by_layers forward method with more examples. In particular more layers (error in commit dd9420e, forward of ann_by_layers applies forward to x on iterations, not y).
				
class test_ann_by_layers(unittest.TestCase):

	def __init__(this, *args, **kwargs):
		# Initialize test attributes and parameters
		
		# Parameter for valid inputs to constructor.
		this.validInputs = [([],''), ([],'test'), ([], 'test')]
		
		# Super constructor.
		super(test_ann_by_layers, this).__init__(*args, **kwargs)

	def test_constructor(this):
	
		# Test the constructor
		
		for layers, name in this.validInputs:
			with this.subTest():
				nn = ann_by_layers(layers,name)
				this.assertIsInstance(nn, ann)
				this.assertEqual(nn.name, name)
				this.assertEqual(nn.layers, layers)

	def test_print(this):
		
		# Test the display of print(ann)
		
		for layers, name in this.validInputs:
			with this.subTest():
				nn = ann_by_layers(layers,name)
				catch_stdout = io.StringIO()
				with contextlib.redirect_stdout(catch_stdout):
					print(nn)
				this.assertEqual(catch_stdout.getvalue(), nn.__class__.__name__ + " class with name " + nn.name + '\n') 
		def test_forward(this):		# Test the forward method of ann_by_layers		l = [layers.fc(NumHidden=10, W = np.eye(10), b = np.zeros((10,1))), layers.relu()]		nn = ann_by_layers(layers = l)		x = np.arange(-5,5)		x = x[:, None]		y = nn.forward(x)		y_exp = x		y_exp[x<0] = 0		nptest.assert_array_equal(y,y_exp)		
if __name__ == '__main__':
	unittest.main()