import unittest
from ann import ann, ann_by_layers
import io
import contextlib

# To Consider/Add:
# 1. Mock test the base class
# 2. Test ann_by_layers with layer array inputs.
				
class test_ann_by_layers(unittest.TestCase):

	def __init__(this, *args, **kwargs):
		# Initialize test attributes and parameters
		
		# Parameter for valid inputs to cosntructor.
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
	
if __name__ == '__main__':
	unittest.main()