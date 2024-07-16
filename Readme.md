### Neural Network 2024 (Rewrite)
This is a rewrite of my proprietary neural network. I was struggling to make backpropagation work in my earlier version due to some mistakes in the beginning of the project, so I began a rewrite of the project with knowledge based on how the last went.
Currently backpropagation by stochastic gradient descent is effective in this project, I am still working on fine tuning it to get it more precise. All machine learning code is my own, I used MathNET Numerics to handle some of the basic vector and matrix math.
I wrote this to gain a further understanding of how deep learning libraries work at their core. Simply running a few PyTorch commands and magically obtaining an optimized network didn't satisfy my curiosity; I wanted to know exactly how perceptron networks work.

## How does it work?
The neural network has two main modes, categorical and linear. The categorical mode is what you're likely used to seeing in traditional machine learning applications. Categorical mode uses Categorical Cross-Entropy loss (CCE) to train the network for categorizing data, in this case the iris setosa dataset. 
Linear mode uses DataGenerator to create a dataset for the neural network to work with out of a function supplied by the user. Program.cs creates a Driver object that contains the neural network and everything the neural network needs to function. The driver constructor is where all the settings of the neural network can be customized.
For saving networks to disk, I use JSON Serialization to convert the objects into file and vice versa.
