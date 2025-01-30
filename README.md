# Machine learning Orbit Prediction

## Prophet + FNN
First part of the project aims to recreate the paper Precise and Efficient Orbit Prediction in LEO with Machine Learning using Exogenous Variables by Francisco Caldas and Cláudia Soares available [here](https://arxiv.org/abs/2407.11026).
In this part dataset was scraped and procesed before use.

The main idea is to use facebook [Prophet](https://facebook.github.io/prophet/) to make coarse predictions on the position of the satekite in orbit based on last 1000 known positions and then use the output and aditional variables for a more accurate prediction. This adjustment of the prediction is performed by a feed-forward neural network that consists of three hidden layers each consisting of 100 units with LeakyReLu as the activation function. 


- Directory data contains datasets prepared for full pipeline that includes prophet and the neural network as well as preprocesed inputs for only the neural network which speed up the training process significantly.
- Directories datasetsProphet and datasetsExogenous include scripts that were used to acquire the coresponding data.
- Directory prophetandfnn include the predictionPipleline as well as preprocessing for nn inputs
- Directory  prophetLarets includes prophet model development scripts mainly the prediction script as well as a script which performs Fast Fourier Transform over the dataset to determine seasonality frequencies in the data

In the end the models perfomance wasn't good enough so another approach was taken.

## LSTM
Second part of the project tries a different approach using a LSTM model to predict the future position of a satelite, using the same data as in the first part of the project. The outcome of this approach proved more favorable (within 0.1% on test data), and was therefore federalised in the third part of the project.


## Federalised LSTM orbit prediction

In the third part of the project(LSTMfed) the LSTM sequential code got federalised to be used in a distributed manner with [PTB-FLA](https://github.com/miroslav-popovic/ptbfla) federated learning framework.


