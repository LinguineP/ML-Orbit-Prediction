from enum import Enum

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import multiprocessing
from tqdm import tqdm

import matplotlib.pyplot as plt

import prophetLarets.predictCoarse  as coarse 
import fnn


def train_predict_for_series(series_data, prophet_model, data, index):
    return coarse.train_predict_1000(prophet_model, data, index)

def parallel_train_predict(data_x, prophetModel_x, data_y, prophetModel_y, data_z, prophetModel_z, index):
    # Create a pool of processes to execute the function in parallel
    with multiprocessing.Pool(processes=3) as pool:
        # Map the function over the three different series data and models
        results = pool.starmap(train_predict_for_series, [
            (data_x, prophetModel_x, data_x, index),
            (data_y, prophetModel_y, data_y, index),
            (data_z, prophetModel_z, data_z, index)
        ])
    
    # Unpack the results from each parallel task
    (timeStamp_x, realValue_x, predictedValue_x, prophetModel_x, prev_timestamp_x) = results[0]
    (timeStamp_y, realValue_y, predictedValue_y, prophetModel_y, prev_timestamp_y) = results[1]
    (timeStamp_z, realValue_z, predictedValue_z, prophetModel_z, prev_timestamp_z) = results[2]
    
    return timeStamp_x, realValue_x, predictedValue_x, prophetModel_x, prev_timestamp_x, \
           timeStamp_y, realValue_y, predictedValue_y, prophetModel_y, prev_timestamp_y, \
           timeStamp_z, realValue_z, predictedValue_z, prophetModel_z, prev_timestamp_z


def fetchProphetData(datasetPath):
    """
        gets data for use within prophet from the right file

    
    """  
    
    in_data = pd.read_csv(datasetPath)

    timeSeriesData = coarse.transform_to_timestamp(in_data)

    data_x, data_y, data_z = coarse.separate_components(timeSeriesData)
    
    return data_x,data_y,data_z


def fetchExogenousData(datasetPath):
    in_data=pd.read_csv(datasetPath)
    return in_data



def load_nn_data(inputs_path, targets_path):
    """
    Function to load the saved inputs and targets tensors from .pt files.
    
    Args:
    - inputs_path (str): Path to the saved inputs tensor file.
    - targets_path (str): Path to the saved targets tensor file.
    
    Returns:
    - inputs (torch.Tensor): Loaded inputs tensor.
    - targets (torch.Tensor): Loaded targets tensor.
    """
    # Load the saved tensors from .pt files
    inputs = torch.load(inputs_path)
    targets = torch.load(targets_path)

    # Return the loaded tensors
    return inputs, targets

def generateNNinputs(inputs_path,targets_path):
    prophetTrainDataPath="/home/pavle/op-ml/data/finalData/train_data.csv"
    exogenousDataPath="/home/pavle/op-ml/data/finalData/mergedExogenous.csv"
    
    
    
    data_x, data_y, data_z=fetchProphetData(prophetTrainDataPath)
    exogenousData=fetchExogenousData(exogenousDataPath)
    
    stopIndex = 1100
    
    prophetModel_x=None
    prophetModel_y=None
    prophetModel_z=None
    
    errdf_x = pd.DataFrame(
        columns=["timeStamp", "realValue", "predictedValue", "error"]
    )
    
    errdf_y = pd.DataFrame(
        columns=["timeStamp", "realValue", "predictedValue", "error"]
    )
    
    
    errdf_z = pd.DataFrame(
        columns=["timeStamp", "realValue", "predictedValue", "error"]
    )
    
    stepSize=240
    
    startIndex=1000
    stopIndex=1240
    numberOfSteps=(len(data_x.index)-startIndex)//stepSize
    

    input_batch=[]
    target_batch=[]
    for step in tqdm(range(numberOfSteps), desc="Training Steps", ncols=100):
        print("Starting prophet model training...")
        start = time.time()
        
        for index, row in data_x.iloc[startIndex:].iterrows():
            if index == stopIndex:
                stopIndex += stepSize
                startIndex += stepSize
                break
            
            # Execute the train_predict_1000 function calls in parallel
            (timeStamp_x, realValue_x, predictedValue_x, prophetModel_x, prev_timestamp_x,
            timeStamp_y, realValue_y, predictedValue_y, prophetModel_y, prev_timestamp_y,
            timeStamp_z, realValue_z, predictedValue_z, prophetModel_z, prev_timestamp_z) = parallel_train_predict(
                data_x, prophetModel_x, data_y, prophetModel_y, data_z, prophetModel_z, index)
            
            # Retrieve the exogenous data
            exogenousState = exogenousData.loc[exogenousData['Timestamp'] == prev_timestamp_x]
            
            try:
                exogenous = exogenousState.drop("Timestamp", axis=1).iloc[0].to_numpy()
            except:
                print(timeStamp_x)
            
            # Create the input batch
            predicted = np.array([predictedValue_x, predictedValue_y, predictedValue_z])
            input = np.concatenate([predicted, exogenous])  # making a single input
            target = [realValue_x, realValue_y, realValue_z]
            
            # Append the results
            input_batch.append(input)
            target_batch.append(target)
            
            
    
    inputs = torch.tensor(np.array(input_batch), dtype=torch.float32)    
    targets = torch.tensor(np.array(target_batch), dtype=torch.float32)
    
    
    torch.save(inputs, inputs_path)
    torch.save(targets, targets_path)
    
if __name__=="__main__":
    inputs_path="/home/pavle/op-ml/nnInputs/inputs.pt"
    targets_path="/home/pavle/op-ml/nnInputs/targets.pt"
    generateNNinputs(inputs_path,targets_path)