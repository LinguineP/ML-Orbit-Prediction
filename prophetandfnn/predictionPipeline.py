from enum import Enum

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import multiprocessing
import pickle
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import prophetLarets.predictCoarse  as coarse 
import fnn


class ErrorType(Enum):
    ABSOLUTE = 1
    RSE = 2



def calculate_error_concat(errorType:ErrorType,realValue,predictedValue,timeStamp,df):
    error=0
    if errorType==ErrorType.ABSOLUTE:
        error = abs(realValue - predictedValue)
    else:
        error= np.sqrt(np.pow((realValue-predictedValue),2))

        
    df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "timeStamp": timeStamp,
                        "realValue": realValue,
                        "predictedValue": predictedValue,
                        "absoluteError": error,
                    },
                    index=[0],
                ),
            ]
        )
    return df,error


def train_predict_for_series( prophet_model, data, index,isZ=False):
    
    return coarse.train_predict_1000(prophet_model, data, index,isZ)

def parallel_train_predict(data_x, prophetModel_x, data_y, prophetModel_y, data_z, prophetModel_z, index):
    # Create a pool of processes to execute the function in parallel
    with multiprocessing.Pool(processes=3) as pool:
        # Map the function over the three different series data and models
        results = pool.starmap(train_predict_for_series, [
            (prophetModel_x, data_x, index),
            (prophetModel_y, data_y, index),
            (prophetModel_z, data_z, index,True)
        ])
    
    # Unpack the results from each parallel task
    (timeStamp_x, realValue_x, predictedValue_x, prophetModel_x, prev_timestamp_x) = results[0]
    (timeStamp_y, realValue_y, predictedValue_y, prophetModel_y, prev_timestamp_y) = results[1]
    (timeStamp_z, realValue_z, predictedValue_z, prophetModel_z, prev_timestamp_z) = results[2]
    
    return timeStamp_x, realValue_x, predictedValue_x, prophetModel_x, prev_timestamp_x, \
           timeStamp_y, realValue_y, predictedValue_y, prophetModel_y, prev_timestamp_y, \
           timeStamp_z, realValue_z, predictedValue_z, prophetModel_z, prev_timestamp_z


def validate(nnModel, startIndex=1000, stopIndex=1100, stepSize=50, criterion=nn.L1Loss()):
    """
    Validate the trained model on validation data and compute the error metrics.
    
    Parameters:
    - nnModel: The trained neural network model.
    - prophetTrainDataPath: Path to the prophet training data (CSV).
    - exogenousDataPath: Path to the exogenous data (CSV).
    - startIndex: The starting index for validation data.
    - stopIndex: The stopping index for validation data.
    - stepSize: The batch size for validation.
    - criterion: Loss function (e.g., nn.MSELoss()).
    
    Returns:
    - val_loss: The calculated validation loss.
    """
    prophetValDataPath="/home/pavle/op-ml/data/finalData/val_data.csv"
    exogenousDataPath="/home/pavle/op-ml/data/finalData/mergedExogenous.csv"
    
    
    # Load the data in the same way as in training
    data_x, data_y, data_z = fetchProphetData(prophetValDataPath)
    exogenousData = fetchExogenousData(exogenousDataPath)
    exogenousData=drop_columns(exogenousData)
    
    nnModel.eval()  # Set the model to evaluation mode
    

    numberOfSteps=(len(data_x.index)-startIndex)//stepSize
    
    numberOfSteps=1
    
    
    val_loss = 0.0
    
    ret_outputs=[]
    ret_targets=[]

    
    nnModel.eval()
    with torch.no_grad():  # Disable gradient computation during validation
        for step in range(numberOfSteps):
            
            input_batch=[]
            target_batch=[]
            print("Starting prophet model training...")
            start = time.time()
            
            prophetModel_x=None
            prophetModel_y=None
            prophetModel_z=None
            
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
                    
                input_row=[predictedValue_x,predictedValue_y,predictedValue_z]+ list(exogenous)
                target_row=[realValue_x,realValue_y,realValue_z]
                input_batch.append(input_row)
                target_batch.append(target_row)    


        inputs = torch.tensor(np.array(input_batch), dtype=torch.float32)
        targets = torch.tensor(np.array(target_batch), dtype=torch.float32)
            
            
            
        
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
        nnModel.to(device)
    
        inputs = inputs.to(device)
        targets = targets.to(device)
    
    
        outputs = nnModel(inputs)
        batch_loss = criterion(outputs, targets)
        val_loss += batch_loss.item()
        ret_targets+=targets.cpu().detach().tolist()
        ret_outputs+=outputs.cpu().detach().tolist()
        
    val_loss /= numberOfSteps  # Calculate average loss across all validation steps
    print(f"Validation Loss: {val_loss}")
    
    
    
    
    return val_loss,ret_outputs,ret_targets
    


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

def plotLoss(values,epoch):
    
    indices = range(len(values))
    

    # Plot the values against their indices and draw a line through them
    plt.plot(indices, values, marker='o', linestyle='-', color='b')

    # Add labels and title
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('loss_progression_epoch')

    # Show the plot
    plt.savefig(f'/home/pavle/op-ml/trainingOutput/loss_progression{epoch}.png')
    
    plt.clf()
    
    plt.show()
    
    
    
def generateNNinputs():
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
    
    stepSize=20
    
    startIndex=1000
    stopIndex=1020
    numberOfSteps=(len(data_x.index)-startIndex)//stepSize
    

    input_batch=[]
    target_batch=[]
    for step in range(numberOfSteps):
            
            
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
    
    
    torch.save(inputs, "nnInputs/inputs.pt")
    torch.save(targets, "nnInputs/targets.pt")
                

def train():
    
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
    
    stepSize=20
    
    startIndex=1000
    stopIndex=1020
    numberOfSteps=(len(data_x.index)-startIndex)//stepSize
    
    nnModel=fnn.ffNNet()
    
    optimizer = optimizer = torch.optim.Adam(nnModel.parameters())

    criterion=nn.MSELoss()
    
    numberOfEpochs=10
    
    
    for ep in range(numberOfEpochs):
        loss_track=[]
        startEp = time.time()
        for step in range(numberOfSteps):
            input_batch=[]
            target_batch=[]
            
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
            
            print("Starting NN training...")    
                
            inputs = torch.tensor(np.array(input_batch), dtype=torch.float32)
            targets= torch.tensor(np.array(target_batch), dtype=torch.float32)   
            end = time.time()
            delta=end-start
            
            
            loss=fnn.train_nn(nnModel,inputs,targets,optimizer,criterion)
            
            
            
            
            print(f"[{delta}] Loss after step [{step+1}/{numberOfSteps}]: {loss}")
            
            loss_track.append(loss)
            
            with open('loss.log', 'a') as file:
                file.write(f"Loss after step [{step+1}/{numberOfSteps}]: {loss}  execTime: {delta}\n")
        
        
        endEp = time.time()
        deltaEp=endEp-startEp
        torch.save({
            'model_state_dict': nnModel.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': ep,
            'loss': loss,
        }, f'checkpoint{ep}.pth')
        print(f"[{deltaEp}] Loss after epoch [{ep+1}/{numberOfEpochs}]: {loss_track[-1]}")        
        plotLoss(loss_track,ep)
    
    return nnModel,optimizer,epoch,loss
        

def drop_columns(df):
    
    """
    columns_to_drop = [
        "Field_Magnitude_nT", "Speed_km_per_s", "Proton_Density_n_per_cc", "Proton_Temperature_K", 
        "Flow_Pressure_nPa", "Alfven_Mach_Number", "Kp_index", "Lyman_alpha", "p", "f", "g", "h", 
        "k", "l", "semi_major_axis_a", "eccentricity_e", "inclination_i", "longitude_of_ascending_node_Omega", 
        "argument_of_perihelion_omega", "true_anomaly_nu", "total_mass_density", "pertrubation_r", 
        "pertrubation_theta", "perturbation_phi"
    ]
    """
    columns_to_drop = [
        "Field_Magnitude_nT", "Speed_km_per_s", "Proton_Density_n_per_cc", "Proton_Temperature_K", 
        "Flow_Pressure_nPa", "Alfven_Mach_Number", "Kp_index", "Lyman_alpha", "semi_major_axis_a", "eccentricity_e", "inclination_i", "longitude_of_ascending_node_Omega", 
        "argument_of_perihelion_omega", "true_anomaly_nu"
    ]
    
    
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    return df

def load_nn_data(inputs_path, targets_path):
    """
    Loads data from input and target CSV files, converts them to Pandas dataframes, 
    and then converts them to PyTorch tensors.

    Args:
        input_path (str): File path for the input CSV.
        target_path (str): File path for the target CSV.
        device (str): The device to place the tensors on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Tensor containing inputs.
        torch.Tensor: Tensor containing targets.
    """
    
    input_dataframe = pd.read_csv(inputs_path)

    if 'timeStamp' in input_dataframe.columns:
        input_dataframe = input_dataframe.drop(columns=['timeStamp'])
        input_dataframe = input_dataframe.drop(columns=['Timestamp'])
        
    input_dataframe=drop_columns(input_dataframe)
    
    target_dataframe = pd.read_csv(targets_path)
    
    if 'timeStamp' in target_dataframe.columns:
        target_dataframe = target_dataframe.drop(columns=['timeStamp'])
        
    
    
    input_dataframe = input_dataframe.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0) #minmax normalisation over the whole dataset
    inputs_tensor = torch.tensor(input_dataframe.values, dtype=torch.float32)
    targets_tensor = torch.tensor(target_dataframe.values, dtype=torch.float32)
    

    return inputs_tensor, targets_tensor


def plotOutputs(ground_truth, predictions): 
    
    
    if True:
        diff_x = ground_truth[:, 0] - predictions[:, 0]
        diff_y = ground_truth[:, 1] - predictions[:, 1]
        diff_z = ground_truth[:, 2] - predictions[:, 2]

        # Create subplots for each difference
        fig, ax = plt.subplots(3, 1, figsize=(10, 12))

        # Plot X vs X (ground truth vs predictions)
        ax[0].scatter(ground_truth[:, 0], predictions[:, 0], color='blue', label="X vs X Predicted", alpha=0.6)
        ax[0].plot([ground_truth[:, 0].min(), ground_truth[:, 0].max()], [ground_truth[:, 0].min(), ground_truth[:, 0].max()], 'r--', label="Ideal")
        ax[0].set_title('Ground Truth X vs Predicted X')
        ax[0].set_xlabel('Ground Truth X')
        ax[0].set_ylabel('Predicted X')
        ax[0].legend()
        ax[0].grid(True)

        # Plot Y vs Y (ground truth vs predictions)
        ax[1].scatter(ground_truth[:, 1], predictions[:, 1], color='green', label="Y vs Y Predicted", alpha=0.6)
        ax[1].plot([ground_truth[:, 1].min(), ground_truth[:, 1].max()], [ground_truth[:, 1].min(), ground_truth[:, 1].max()], 'r--', label="Ideal")
        ax[1].set_title('Ground Truth Y vs Predicted Y')
        ax[1].set_xlabel('Ground Truth Y')
        ax[1].set_ylabel('Predicted Y')
        ax[1].legend()
        ax[1].grid(True)

        # Plot Z vs Z (ground truth vs predictions)
        ax[2].scatter(ground_truth[:, 2], predictions[:, 2], color='red', label="Z vs Z Predicted", alpha=0.6)
        ax[2].plot([ground_truth[:, 2].min(), ground_truth[:, 2].max()], [ground_truth[:, 2].min(), ground_truth[:, 2].max()], 'r--', label="Ideal")
        ax[2].set_title('Ground Truth Z vs Predicted Z')
        ax[2].set_xlabel('Ground Truth Z')
        ax[2].set_ylabel('Predicted Z')
        ax[2].legend()
        ax[2].grid(True)

        # Show the plots
        plt.tight_layout()
        plt.show()
        
    n_samples=400
    ground_truth=ground_truth[100:n_samples]
    predictions=predictions[100:n_samples]
    
    if True:
        fig, ax = plt.subplots(2, 3, figsize=(15, 10))

        # Plot X (ground truth) vs X (predicted)
        ax[0, 0].plot(ground_truth[:, 0], label='Ground Truth X', color='blue', linewidth=2)
        ax[0, 0].plot(predictions[:, 0], label='Predicted X', color='lightblue', linestyle='--', linewidth=2)
        ax[0, 0].set_title('Ground Truth X vs Predicted X')
        ax[0, 0].set_xlabel('Sample Index')
        ax[0, 0].set_ylabel('X Values')
        ax[0, 0].legend()

        # Plot Y (ground truth) vs Y (predicted)
        ax[0, 1].plot(ground_truth[:, 1], label='Ground Truth Y', color='green', linewidth=2)
        ax[0, 1].plot(predictions[:, 1], label='Predicted Y', color='lightgreen', linestyle='--', linewidth=2)
        ax[0, 1].set_title('Ground Truth Y vs Predicted Y')
        ax[0, 1].set_xlabel('Sample Index')
        ax[0, 1].set_ylabel('Y Values')
        ax[0, 1].legend()

        # Plot Z (ground truth) vs Z (predicted)
        ax[0, 2].plot(ground_truth[:, 2], label='Ground Truth Z', color='red', linewidth=2)
        ax[0, 2].plot(predictions[:, 2], label='Predicted Z', color='lightcoral', linestyle='--', linewidth=2)
        ax[0, 2].set_title('Ground Truth Z vs Predicted Z')
        ax[0, 2].set_xlabel('Sample Index')
        ax[0, 2].set_ylabel('Z Values')
        ax[0, 2].legend()

        # Plot X (ground truth)
        ax[1, 0].plot(ground_truth[:, 0], label='Ground Truth X', color='blue', linewidth=2)
        ax[1, 0].set_title('Ground Truth X')
        ax[1, 0].set_xlabel('Sample Index')
        ax[1, 0].set_ylabel('X Values')
        ax[1, 0].legend()

        # Plot Y (ground truth)
        ax[1, 1].plot(ground_truth[:, 1], label='Ground Truth Y', color='green', linewidth=2)
        ax[1, 1].set_title('Ground Truth Y')
        ax[1, 1].set_xlabel('Sample Index')
        ax[1, 1].set_ylabel('Y Values')
        ax[1, 1].legend()

        # Plot Z (ground truth)
        ax[1, 2].plot(ground_truth[:, 2], label='Ground Truth Z', color='red', linewidth=2)
        ax[1, 2].set_title('Ground Truth Z')
        ax[1, 2].set_xlabel('Sample Index')
        ax[1, 2].set_ylabel('Z Values')
        ax[1, 2].legend()

        # Show the plots
        plt.tight_layout()
        plt.show()
        
    
    if(False):
        # Create figure
        fig = plt.figure(figsize=(14, 6))

        # Subplot for Ground Truth
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], c='b', label='Targets')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Ground Truth (Targets)')
        ax1.legend()

        # Subplot for Predictions
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(predictions[:, 0], predictions[:, 1], predictions[:, 2], c='r', label='Predictions')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('Model Predictions')
        ax2.legend()

        # Display the plot
        plt.show()

def add_noise_to_params(model, noise_factor=0.01):
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * noise_factor
            param.add_(noise)

def train_no_prophet():
    inputs_path = "/home/pavle/op-ml/nnInputs/inputsFull.csv"
    targets_path = "/home/pavle/op-ml/nnInputs/targetsFull.csv"

    inputs, targets = load_nn_data(inputs_path, targets_path)
    input_size = inputs[0].size(0) 

    load_model=True

    nnModel = fnn.ffNNet(input_size)
    optimizer = torch.optim.Adam(nnModel.parameters(), lr=1e-2)
    criterion = nn.L1Loss()
    scheduler = None #torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
    numberOfEpochs = 50000
    
    ground_truth=None
    outputs=None
    
    

    if load_model:
        
        
        checkpoint = torch.load("/home/pavle/op-ml/trainingOutput/bestmodelFullIn.pth",weights_only=True)
    
        
        nnModel.load_state_dict(checkpoint['model_state_dict'])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        nnModel = nnModel.to(device)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        
    


    # Load inputs and targets from saved .pt files
    inputs, targets = load_nn_data(inputs_path, targets_path)
    loss_track = []
    loss_cur=0
    loss1=[]
    
    
    epochs_without_improvement = 0
    patience = 1000
    best_loss=np.inf
    best_model=None
    for ep in range(numberOfEpochs):
        
        startEp = time.time()

        
        batch_size = len(inputs)  
        num_batches = (len(inputs) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(inputs))

            batch_inputs = inputs[start_idx:end_idx]
            batch_targets = targets[start_idx:end_idx]

            # Train the NN
            start = time.time()
            ground_truth=batch_targets
            loss,outputs= fnn.train_nn(nnModel, batch_inputs, batch_targets, optimizer, criterion,scheduler=scheduler)
            end = time.time()

            delta = end - start
            

            print(f"[{delta}] Loss after batch [{batch_idx + 1}/{num_batches}]: {loss}")
            with open('loss.log', 'a') as file:
                file.write(f"Loss after batch [{batch_idx + 1}/{num_batches}]: {loss}  execTime: {delta}\n")
        loss_cur=loss
        ignorePercent=30 #controls how much of the last % of the graph will be shown  0= wholle graph is shown 100 nothing is shown
        if ep>((numberOfEpochs//100)*ignorePercent):
            loss_track.append(loss)
        loss1.append(loss)
        endEp = time.time()
        deltaEp = endEp - startEp
        
        
    
        
        
        

        # Save model checkpoint
        
    
        print(f"[{deltaEp}] Loss after epoch [{ep + 1}/{numberOfEpochs}]: {loss_cur}")
    
    

    torch.save({
            'model_state_dict': nnModel.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': ep,
            'loss': loss,
    }, f'/home/pavle/op-ml/trainingOutput/checkpoint{ep}.pth')
    
    ground_truth = ground_truth.cpu().detach()
    outputs = outputs.cpu().detach()
    inputs = inputs.cpu().detach()
    
    #_,outputs,targets=validate(nnModel)
    
    #print(outputs)
    
    #plotOutputs(np.array(outputs),np.array(targets))
    plotLoss(loss_track, ep)
    plotLoss(loss1, ep+1)
    plotOutputs(targets.numpy(),outputs.numpy()) #nonvalidation
    
    
    

    return nnModel, optimizer, ep, loss




if __name__ == "__main__":
    

    nnModel,optimizer,epoch,loss=train_no_prophet()
    