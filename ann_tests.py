import unittest
from ann import ann, ann_by_layers, mlp
import io
import contextlibimport layersimport numpy.testing as nptestimport numpy as np

# To Consider/Add:
# 1. Mock test the base class
# 2. Test ann_by_layers with layer array inputs - maybe mock layers once/if input validation added.# 3. Test ann_by_layers forward method with more examples. In particular more layers (error in commit dd9420e, forward of ann_by_layers applies forward to x on iterations, not y).# 4. Test equality overload?
				
class test_ann_by_layers(unittest.TestCase):

	def __init__(this, *args, **kwargs):
		# Initialize test attributes and parameters
		
		# Parameter for valid inputs to constructor.
		this.validInputs = [([],''), ([],'test'), ([], 'test')]		# Options for initialize_training_parameters		this.initializerArguments = { 
			'Default' : {},
			'Custom' : { np.random.uniform }
			}
		
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
		def test_forward(this):		# Test the forward method of ann_by_layers		l = [layers.fc(InputSize = 10, NumHidden=10, W = np.eye(10), b = np.zeros((10,1))), layers.relu()]		nn = ann_by_layers(layers = l)		x = np.arange(-5,5)		x = x[:, None]		y = nn.forward(x)		y_exp = x		y_exp[x<0] = 0		nptest.assert_array_equal(y,y_exp)	def test_forward_to_layer(this):		# Test the forward_to_layer convenience method.
		l = [layers.fc(InputSize = 10, NumHidden=10, W = np.eye(10), b = -1*np.ones((10,1))), layers.relu()]
		nn = ann_by_layers(layers = l)
		x = np.arange(-5,5)
		x = x[:, None]		y = nn.forward_to_layer(x,1)		nptest.assert_array_equal(y, x-np.ones((10,1)))	def test_collect_forwards(this):		l = [layers.fc(InputSize = 5, NumHidden = 10), layers.relu(), layers.fc(InputSize = 10, NumHidden = 3)]
		nn = ann_by_layers(layers=l)		nn.initialize_training_parameters()		x = np.random.random(size = (5,1))		forwards = nn.collect_forwards(x)		# we collect forwards, including before the first layer (i.e. the identity)		this.assertEqual(len(forwards), len(nn.layers)+1)		y = x		nptest.assert_array_equal(y, forwards[0])		for i in range(3):			y = nn.layers[i].forward(y)			nptest.assert_array_equal(y,forwards[i+1])				def test_collect_mid_layer_gradients(this):		l = [layers.fc(InputSize = 5, NumHidden = 10), layers.relu(), layers.fc(InputSize = 10, NumHidden = 3)]
		nn = ann_by_layers(layers=l)
		nn.initialize_training_parameters()
		x = np.random.random(size = (5,1))		dxMidLayerCollection = nn.collect_mid_layer_gradients(x)		this.assertEqual(len(dxMidLayerCollection), len(nn.layers)+1)		dldx = np.eye(3)		for i in range(3):			xi = nn.forward_to_layer(x,i)			dldx = np.matmul(dldx, nn.layers[2-i].x_gradient(xi))		nptest.assert_array_equal(dldx, dxMidLayerCollection[0])	def test_backprop_updates(this):		l = [layers.fc(InputSize = 5, NumHidden = 10), layers.relu(), layers.fc(InputSize = 10, NumHidden = 3)]
		nn = ann_by_layers(layers=l)
		nn.initialize_training_parameters()
		x = np.random.random(size = (5,1))		y = np.random.random(size = (3,1))		loss = layers.mse()		# setting learn rate to 0 should kill all gradients.		updates = nn.backprop_updates(x, y, loss, lr = 0)		params = nn.get_training_parameters(x)		for i,update in zip(range(len(updates)),updates):			for param in update:				dLdp = update[param]['Update']				thisParamName = params[i][param]['Name']				thisParam = getattr(nn.layers[i], thisParamName)				nptest.assert_array_equal(dLdp, np.zeros_like( thisParam ))		def test_update_training_parameters(this):		# Test the update_training_parameters method		l = [layers.fc(InputSize = 5, NumHidden = 10), layers.relu(), layers.fc(InputSize = 10, NumHidden = 3)]
		nn = ann_by_layers(layers=l)
		nn.initialize_training_parameters()
		x = np.random.random(size = (5,1))
		y = np.random.random(size = (3,1))
		loss = layers.mse()
		params = nn.get_training_parameters()		updates = []		for i,layer_params in zip(range(len(nn.layers)),params):			layer_updates = {}			for param in layer_params:				thisParamName = layer_params[param]['Name']				thisParamValue = getattr(nn.layers[i], thisParamName)				layer_updates[param] = {}				layer_updates[param]['Update'] = thisParamValue				layer_updates[param]['Name'] = thisParamName			updates.append(layer_updates)		nn.update_training_parameters(updates)		for layer in nn.layers:			params = layer.get_training_parameters()			for param in params:				paramValue = getattr(layer, params[param]['Name'])				nptest.assert_array_equal(paramValue, np.zeros_like( paramValue ))						def test_initialize_training_parameters(this):		# Test the initialize_training_parameters method of ann_by_layers		l = [layers.fc(InputSize = 5, NumHidden = 10), layers.relu(), layers.fc(InputSize = 10, NumHidden = 3)]		nn = ann_by_layers(layers=l)		for initializerArg in this.initializerArguments:
			with this.subTest():				nn.initialize_training_parameters(*this.initializerArguments[initializerArg])				this.assertEqual(nn.layers[0].W.shape, (10, 5))				this.assertEqual(nn.layers[2].b.shape, (3, 1) )	def test_get_training_parameters(this):		# Test the get_training_parameters method of ann_by_layers		l = [layers.fc(InputSize = 5, NumHidden = 10), layers.relu(), layers.fc(InputSize = 10, NumHidden = 3)]
		nn = ann_by_layers(layers=l)		params = nn.get_training_parameters()		for i in range(len(l)):			this.assertEqual(params[i], l[i].get_training_parameters())		nn.initialize_training_parameters()			x = np.arange(5)		x = x[:,None]		params = nn.get_training_parameters(x)		layer_params = []		for i in range(len(l)):			layer_params.append(l[i].get_training_parameters(x))			x = l[i].forward(x)		nptest.assert_array_equal(params[0]['Weights']['Gradient'], layer_params[0]['Weights']['Gradient'])class test_mlp(unittest.TestCase):	def test_constructor(this):		nn = mlp(InputSize = 5, HiddenSizes = [10,20,3])		l = [layers.fc(InputSize = 5, NumHidden = 10), layers.relu(), layers.fc(InputSize = 10, NumHidden = 20), layers.relu(),				layers.fc(InputSize = 20, NumHidden = 3), layers.relu()]		nn_exp = ann_by_layers(layers = l)		this.assertEqual(nn.cast_to_ann_by_layers(), nn_exp)
if __name__ == '__main__':
	unittest.main()