import unittest
from ann import ann
import io
import contextlib

class test_ann(unittest.TestCase):

	def __init__(this, *args, **kwargs):
		# Initialize test attributes and parameters
		
		# Parameter for valid inputs to cosntructor.
		this.validInputs = [([],''), ([],'test'), ([], 'test')]
		
		# Super constructor.
		super(test_ann, this).__init__(*args, **kwargs)

	def test_constructor(this):
	
		# Test the constructor
		
		for layers, name in this.validInputs:
			with this.subTest():
				nn = ann(layers,name)
				this.assertIsInstance(nn, ann)
				this.assertEqual(nn.name, name)
				this.assertEqual(nn.layers, layers)

	def test_print(this):
		
		# Test the display of print(ann)
		
		for layers, name in this.validInputs:
			with this.subTest():
				nn = ann(layers,name)
				catch_stdout = io.StringIO()
				with contextlib.redirect_stdout(catch_stdout):
					print(nn)
				this.assertEqual(catch_stdout.getvalue(), nn.__class__.__name__ + " class with name " + nn.name + '\n') 
	
if __name__ == '__main__':
	unittest.main()