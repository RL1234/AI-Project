# Testing Various AI Models for Game Play
This project uses TensorFlow to train self-play of a game (Space Invaders). 

It will first use a single layer perceptron. 

Then a multi-layer perceptron.

Then, a convolutional neural network. 

Then, a recurrent neural network.

Then, re-enforced learning and Q-learning. 

It is written in python.

## Training Data

**get_training_data.py**

Input - 110 x 134 grayscale pixel screen grab of the game screen

Output - One hot 1) left, 2) space (fire), 3) right, 4) do nothing

**clean_training_data.py**

Identifies and purges bad captures.

**utilities.py**

Utility functions that analyze the captured game play.

**view_training_data.py**

Plays the captured game play while displaying statistics.

## Single Layer Perceptron Neural Network

**train.py**

14,740 input nodes (flattened 110 x 134 image)

4 output nodes

With approximately 35,000 sample, running batch training (100 sample) with gradient descent (.05) and 2000 Epochs, I got an an accuracy of .985294.

Weights for "left" after training:

![alt txt](https://github.com/RL1234/AI-Project/blob/master/figure_1.png)

Weights for "space (fire)" after training:

![alt txt](https://github.com/RL1234/AI-Project/blob/master/figure_1-1.png)

## Multilayer Perceptron Neural Network

**train2.py**

14,740 input nodes (flattened 110 x 134 image)

100 inner layer nodes

4 output nodes

## Convolutional Neural Network

**train3.py**

A work in progress.


