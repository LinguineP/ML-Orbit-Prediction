# Machine Learning Orbit Prediction

## Prophet + FNN  
The first part of the project aims to recreate the paper *Precise and Efficient Orbit Prediction in LEO with Machine Learning using Exogenous Variables* by Francisco Caldas and Cláudia Soares, available [here](https://arxiv.org/abs/2407.11026).  
In this part, the dataset was scraped and processed before use.

The main idea is to use Facebook [Prophet](https://facebook.github.io/prophet/) to make coarse predictions on the position of the satellite in orbit based on the last 1000 known positions and then use the output and additional variables for a more accurate prediction. This adjustment of the prediction is performed by a feed-forward neural network that consists of three hidden layers, each consisting of 100 units, with LeakyReLU as the activation function.

- The **data** directory contains datasets prepared for the full pipeline, which includes Prophet and the neural network, as well as preprocessed inputs for only the neural network, which speeds up the training process significantly.
- The **datasetsProphet** and **datasetsExogenous** directories include scripts that were used to acquire the corresponding data.
- The **prophetandfnn** directory contains the prediction pipeline as well as preprocessing for NN inputs.
- The **prophetLarets** directory includes Prophet model development scripts, mainly the prediction script, as well as a script that performs Fast Fourier Transform over the dataset to determine seasonality frequencies in the data.

In the end, the model's performance wasn't good enough, so another approach was taken.

## LSTM  
The second part of the project tries a different approach using an LSTM model to predict the future position of a satellite, using the same data as in the first part of the project. The outcome of this approach proved more favorable (within 0.1% on test data) and was therefore federated in the third part of the project.

## Federated LSTM Orbit Prediction  
In the third part of the project (LSTMfed), the LSTM sequential code was federated to be used in a distributed manner with the [PTB-FLA](https://github.com/miroslav-popovic/ptbfla) federated learning framework.
