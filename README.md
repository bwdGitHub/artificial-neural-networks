# ANNIE
## (Artificial Neural Networks Implemented Egregiously)
### Intro
This is an old mini-project where I attempted to implement a basic learning/training algorithm in Python + NumPy alone.

This [Medium article](https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6) convinced me it wouldn't be too hard to do something basic like implement a [multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron) (though I used [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) instead of a [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) activation).

There is an example in "interactive testing.ipynb" that suggests training an MLP (with ReLU activations) is working.

I don't intend to update this repo. I might try implementing the more flexible "computational graph" and backpropagation algorithm described in [Deep Learning Book, Chapter 6](https://www.deeplearningbook.org/contents/mlp.html) at some point, probably in a separate repo.

### Disclaimer
Nothing here is guaranteed to be correct, extensible or efficient, though I did make reasonable, some, and no effort to ensure this, respectively.

### Requirements
I'm using :
* Python 3.6.6
* NumPy 1.15.0

for the implementation and 
* matplotlib 2.2.2
in the "interactive testing" notebook for a simple plot.

If something doesn't work inside and outside of the example in "interactive testing", it's almost certainly my fault.

### Code
There are two source files and two test files.

* `layers.py`
* `layers_tests.py`
* `ann.py`
* `ann_tests.py`

The `layers.py` file defines an abstract `layer` class and three standard layers `fc`,`relu` and `mse`. The layers have to define
* a `forward` method
* a `get_training_parameters` method that returns some container of learnable parameters for that layer
* an `initialize_training_parameters` method that chooses initial values for the learnable parameters.

I didn't get round to making a good API for the learnable parameter container, it is simply a dictionary that maps a nice name like 'Weights' or 'Bias' to another dictionary that contains the variable name used for the corresponding property, and in the case `layer.get_training_parameters(x)` is called with a non-empty `x`, the dictionary contains an entry 'Gradient' that contains the value of the gradient of this layer with respect to the corresponding learnable parameter and layer input `x`. I didn't say it wasn't convoluted. To make it clearer:

Given a layer `L` with a learnable parameter `W`, typically called "Weights", suppose `x` is the single input to `L`. Then `L.get_training_parameters(x)['Weights'].Name` is `W` and `L.get_training_parameters(x)['Weights'].Gradient` is the derivative `dL/dW` evaluated at `x`.

The `ann.py` file defines an abstract `ann` class, and implements `ann_by_layers` as a subclass, and `mlp` as a further subclass of that. I viewed `ann` as a container for a directed graph of `layer` objects, in particular the `ann.forward` method was intended to follow the edges of this graph and apply the `layer.forward` method at each node. An `ann` implementation has to specify:

* `forward` - the network level forward propagation
* `initialize_training_parameters` - to call the corresponding method on every layer
* `get_training_parameters` - a method to collect training parameters from each layer into some container.

I only tried implementing this in the simple "sequential" case called `ann_by_layers`. In this case each layer has at most one input and one output connection, and the graph is a single connected component. The container for `get_training_parameters` can then simply be a flat list/array since there is a natural ordering of the layers in the graph. Of course on top of the abstract `ann` interface I also had to add methods for the gradients.

The `mlp` class is a convenience class for constructing an `ann_by_layers` consisting of layers `fc` then `relu` repeatedly for some given "depth".

#### Tests
The tests `layer_tests.py` and `ann_tests.py` were passing last time I checked. In this case CI stands for "Check Intermittently".