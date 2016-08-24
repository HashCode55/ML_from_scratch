"""
Driver program to test the neural network 
"""

import NeuralNetSGD
import mnsit_load

#get the training data
#training data is itself a tuple containing inputs and the outputs
training_data, _, test_data= mnsit_load.load_data_wrapper()

net = NeuralNetSGD.NeuralNet([784, 30, 10])
net.stochastic_gradient(training_data, 30, 10, 3.0, test_data = test_data)
