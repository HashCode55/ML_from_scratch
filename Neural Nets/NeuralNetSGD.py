"""
A simple program implementing neural nets from scratch.
Using MNSIT dataset for testing. 
"""
import random 

#third party library
import numpy as np

class NeuralNet(object):
	def __init__(self, sizes):
		"""
		Class constructor 
		params : sizes is the array containing the neurons in each 
		layer.
		"""
		self.num_layers = len(sizes)
		self.sizes = sizes
		#assigning random biases 
		#now just try to visualise the neural network....
		self.biases = [np.random.randn(x, 1) for x in sizes[1:]]
		self .weights = [np.random.randn(y, x) 
			for x, y in zip(sizes[:-1], sizes[1:])]


	def feed_forward(self, a):
			
		"""
		This function is used to feedforward the neural net
		params : a is the list of activations
		"""	
		for b, w in zip(self.biases, self.weights) :
			a = sigmoid(np.dot(w, a) + b)
		#a is the output	
		return a

	def stochastic_gradient(self, training_data, epochs, mini_batch_size, 
		eta, test_data = None):
		"""
		This is the power of the neural network.
		What it does it learns the weights and the 
		biases through the stochastic gradient method.

		params : training_data - the training data  
			   epochs - the number of times we need to train the neural net
			   mini_batch_size - size of the randomly chosen batch
			   eta - the learning rate 	
		"""
		if test_data: n_test = len(test_data)
		n = len(training_data)
		#epoch is one net pass in the neural network 
		for epoch in xrange(epochs):
			#shuffle the training data 
			random.shuffle(training_data)
			mini_batches = [training_data[k : k + mini_batch_size] for k in xrange(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_params(mini_batch, eta)
			if test_data:
				print "Epoch {0}: {1} / {2}".format(
                    				epoch, self.evaluate(test_data), n_test)				
                    		

	def update_params(self, mini_batch, eta):
		"""
		Another core of the neural network
		Update the weights and the biases 
		params : mini_batch - the random picked up training data
			   eta - learning rate 
		"""
		#the initial biases and the weights 
		ini_bias = [np.zeros(b.shape) for b in self.biases]
		ini_weights = [np.zeros(w.shape) for w in self.weights]
		#x and y are the features and the labels resp.
		#for each training example, do the work 
		for x, y in mini_batch:
			#biases after the back propogation
			back_bias, back_weights = self.backpropogate(x, y)
			#update the arrays 
			ini_bias = [ib + bb for ib, bb in zip(ini_bias, back_bias)]
			ini_weights = [iw + wb for iw, wb in zip(ini_weights, back_weights)]
		#now update the biases and the weights 	
		self.biases = [b - (eta / len(mini_batch)) * ib 
				for b, ib in zip(self.biases, ini_bias)]	
		self.weights = [w - (eta / len(mini_batch)) * wb 
				for w, wb in zip(self.weights, ini_weights)]	


	#################THESE FUNCTIONS ARE SHAMELESSLY COPIED#################

	def backpropogate(self, x, y):
	        	"""
		Algorithm - 
			-> Set the activation of the first layer as the input 
			-> Feed Forward 
			-> Compute the output error using the 1st back propogation eqution (using sigmoid prime) 
			-> Back propogate the error using the 2nd equation
			-> Get the gradient of the cost function using 3rd and the 4th equation 

	        	Return a tuple ``(nabla_b, nabla_w)`` representing the
	       	gradient for the cost function C_x.  ``nabla_b`` and
	       	 ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
	        	to ``self.biases`` and ``self.weights``."""

	        	nabla_b = [np.zeros(b.shape) for b in self.biases]
	        	nabla_w = [np.zeros(w.shape) for w in self.weights]
	        	###STEP-1###
	        	activation = x
	        	activations = [x] # list to store all the activations, layer by layer
	        	zs = [] # list to store all the z vectors, layer by layer
	        	###STEP-2###
	        	for b, w in zip(self.biases, self.weights):
	        		z = np.dot(w, activation) + b 
	        		zs.append(z)
	        		activation = sigmoid(z)
	        		activations.append(activation)

	        	###STEP-3###
	        	delta = self.cost_derivative(activations[-1], y) * \
	        	    sigmoid_prime(zs[-1])
	        	###STEP-4###    
	        	nabla_b[-1] = delta
	        	nabla_w[-1] = np.dot(delta, activations[-2].transpose())
	        	# Note that the variable l in the loop below is used a little
	        	# differently to the notation in Chapter 2 of the book.  Here,
	        	# l = 1 means the last layer of neurons, l = 2 is the
	        	# second-last layer, and so on.  It's a renumbering of the
	        	# scheme in the book, used here to take advantage of the fact
	        	# that Python can use negative indices in lists.
	        	###STEP-5###
	        	for l in xrange(2, self.num_layers):
	        	    z = zs[-l]
	        	    sp = sigmoid_prime(z)
	        	    delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
	        	    nabla_b[-l] = delta
	        	    nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
	        	return (nabla_b, nabla_w)		

	def cost_derivative(self, output_activations, y):
	        	"""Return the vector of partial derivatives \partial C_x /
	        	\partial a for the output activations."""
	        	return (output_activations-y)

	def evaluate(self, test_data):
	        	"""Return the number of test inputs for which the neural
	        	network outputs the correct result. Note that the neural
	        	network's output is assumed to be the index of whichever
	        	neuron in the final layer has the highest activation."""
	        	test_results = [(np.argmax(self.feed_forward(x)), y)
	          	              for (x, y) in test_data]
	        	return sum(int(x == y) for (x, y) in test_results)	 

	################## END END END OF COPIED FUNCTIONS#####################   

def sigmoid(z):
	#z will be a vector 
	return 1.0 / (1.0 + np.exp(-z))
def sigmoid_prime(z):
    	"""Derivative of the sigmoid function."""
    	return sigmoid(z)*(1-sigmoid(z))
