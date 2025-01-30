import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset







def transform_to_timestamp(in_data):
    df = pd.DataFrame(in_data)
    df["Timestamp"] = df["date"] + " " + df["time"]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S")
    df["Timestamp"] = df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df.drop(["date", "time"], axis=1)
    df = df[["Timestamp", "x", "y", "z"]]
    return df



class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout_prob)
        
        # Smoothing layer (fully connected layer to smooth output)
        self.smoothing_layer = nn.Linear(hidden_dim, hidden_dim)  # Smooth using a linear layer
        self.leaky_relu = nn.ReLU()
        
        
        # Fully connected layer for final prediction
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None, c0=None):
        # If hidden and cell states are not provided, initialize them as zeros
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward pass through LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Apply dropout to the output of LSTM (before feeding it to smoothing layer)
        out = self.dropout(out[:, -1, :])  # Selecting the last output and applying dropout
        
        # Apply smoothing layer (to reduce fluctuations)
        x = self.smoothing_layer(out)
        self.leaky_relu = nn.LeakyReLU()
        x = self.smoothing_layer(x)
        x = self.smoothing_layer(x)
        
        # Pass through the fully connected layer for final output
        out = self.fc(x)
        
        return out, hn, cn






def create_sequences_multivariate(data, seq_length, normalize=True):
    xs = []
    ys = []
    
    # Normalize the data if the flag is set to True
    if normalize:
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data)
    else:
        normalized_data = data
    
    # Create the sequences
    for i in range(len(normalized_data) - seq_length):
        x = normalized_data[i:i + seq_length, :]  # Input sequence
        y = normalized_data[i + seq_length, :3]   # Target sequence (first 3 columns)
        xs.append(x)
        ys.append(y)
    
    # Convert to numpy arrays
    xs = np.array(xs)
    ys = np.array(ys)
    
    # Return the sequences and the scaler (if normalization is applied)
    if normalize:
        return xs, ys, scaler
    else:
        return xs, ys


def validate(validation_data, model, criterion, device, batch_size=256):
    model.eval()  # Set the model to evaluation mode
    val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation for validation
        for batch_X, batch_Y in val_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            # Forward pass
            outputs, _, _ = model(batch_X)
            
            # Compute loss
            loss = criterion(outputs, batch_Y)
            total_loss += loss.item()

    # Calculate average loss
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def train(train_data,num_epochs = 5):
    print("Initializing model...")

    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model and move it to the GPU if available
    model = LSTMModel(input_dim=17, hidden_dim=70, layer_dim=3, output_dim=3, dropout_prob=0.2).to(device)
    
    
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001) #optim
    """
    # Data preparation
    columns_to_use = [
        'x', 'y', 'z', 'Field_Magnitude_nT', 'Speed_km_per_s', 'Proton_Density_n_per_cc',
        'Proton_Temperature_K', 'Flow_Pressure_nPa', 'Alfven_Mach_Number', 'Kp_index',
        'Lyman_alpha', 'p', 'f', 'g', 'h', 'k', 'l', 'semi_major_axis_a',
        'eccentricity_e', 'inclination_i', 'longitude_of_ascending_node_Omega',
        'argument_of_perihelion_omega', 'true_anomaly_nu', 'total_mass_density',
        'pertrubation_r', 'pertrubation_theta', 'pertrubation_phi'
    ]
    """
    columns_to_use = [
        'x', 'y', 'z', 'Field_Magnitude_nT', 'Speed_km_per_s', 'Proton_Density_n_per_cc',
        'Proton_Temperature_K', 'Flow_Pressure_nPa', 'Alfven_Mach_Number', 'Kp_index',
        'Lyman_alpha','total_mass_density',
        'pertrubation_r', 'pertrubation_theta', 'pertrubation_phi'
    ]
    
    
    
    data = train_data[columns_to_use].values
    
    
    time_index = np.arange(0, len(data))  # Create time index
    period = 0.073433 * 24 * 60  # Convert period to minutes (0.073433 days in minutes)
    sine_time = np.sin(2 * np.pi * time_index / period)  # Sine transformation
    cosine_time = np.cos(2 * np.pi * time_index / period)  # Cosine transformation
    
    # Add time features to the data
    
    data_with_time_features = np.column_stack((data, sine_time, cosine_time))
    
    
    seq_length = 175 #initial 175(5 days) scaled up to 245(7days)
    X, y, sclr = create_sequences_multivariate(data_with_time_features, seq_length)
    
    

    # Convert data to PyTorch tensors and move to GPU if available
    trainX = torch.tensor(X, dtype=torch.float32).to(device)
    trainY = torch.tensor(y, dtype=torch.float32).to(device)

    print(f"trainX shape: {trainX.shape}")  # (num_samples, seq_length, num_features)
    print(f"trainY shape: {trainY.shape}")  # (num_samples, output_dim)

    # Training loop
    
    h0, c0 = None, None  # Initialize hidden and cell states
    
    plt.ion()  # Turn on interactive mode for real-time plotting
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))  # Create two subplots: one for loss and one for predicted vs expected

    # Loss plot
    loss_line, = ax[0].plot([], [], marker='o', color='b')  # Empty plot for loss initially
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss vs Epochs')
    ax[0].grid(True)

    # Predicted vs Expected plot for last 100 samples
    predictions_line, = ax[1].plot([], [], marker='o', color='r', label='Predictions')
    targets_line, = ax[1].plot([], [], marker='x', color='g', label='Targets')
    ax[1].set_xlabel('Sample')
    ax[1].set_ylabel('Value')
    ax[1].set_title('Predictions vs Expected (Last 100 Samples)')
    ax[1].legend()
    ax[1].grid(True)

    losses = []  # To store loss values
    last_epoch_predictions = []  # To store predictions for the last epoch
    last_epoch_targets = []  # To store targets for the last epoch

    batch_size = 256  # Adjust to a value that fits your GPU 256 for initial scaled down to 32
    dataset = TensorDataset(trainX, trainY)
    
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # The rest for validation

    # Sequential split
    train_dataset = TensorDataset(trainX[:train_size], trainY[:train_size])
    val_dataset = TensorDataset(trainX[train_size:], trainY[train_size:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Starting training
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()
            outputs, h0, c0 = model(batch_X)  # Don't use h0, c0 across batches unless needed
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()

            if epoch == num_epochs - 1:
                last_epoch_predictions.append(outputs.cpu().detach().numpy())
                last_epoch_targets.append(batch_Y.cpu().detach().numpy())

        losses.append(loss.item())
        loss_line.set_data(range(1, epoch + 2), losses)  # Update the plot with the new loss
        ax[0].relim()  # Recalculate the limits of the plot
        ax[0].autoscale_view()  # Autoscale the view to fit new data

        if (epoch + 1) % 5 == 0:
            plt.pause(0.1)  # Pause to update the plot
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            validation_data = TensorDataset(batch_X, batch_Y)  # Replace valX, valY with your validation data
            validation_loss = validate(val_dataset, model, criterion, device, batch_size=256)
            print(f"Validation Loss: {validation_loss:.4f}")

    # Collect last 100 samples from the last epoch
    last_epoch_predictions_flat = [item for sublist in last_epoch_predictions for item in sublist]
    last_epoch_targets_flat = [item for sublist in last_epoch_targets for item in sublist]

    # Get the last 100 samples
    num_samples = 100
    last_predictions = last_epoch_predictions_flat[-num_samples:]
    last_targets = last_epoch_targets_flat[-num_samples:]

    # Ensure the shape is (100, 3) for both predictions and targets
    last_predictions = np.array(last_predictions)
    last_targets = np.array(last_targets)

    

    # Extract x, y, and z values for plotting
    last_predictions_x = last_predictions[:, 0]
    last_predictions_y = last_predictions[:, 1]
    last_predictions_z = last_predictions[:, 2]

    last_targets_x = last_targets[:, 0]
    last_targets_y = last_targets[:, 1]
    last_targets_z = last_targets[:, 2]

    # Update the plot for predictions vs expected values (plot x, y, z separately)
    show_real_values = True  # Set to False if you don't want to show real (target) values

# Update the plot for predictions vs expected values (plot x, y, z separately)
    predictions_line.set_data(range(1, num_samples + 1), last_predictions_x)  # Plot x values
    ax[1].plot(range(1, num_samples + 1), last_predictions_y, label="Predicted Y", color="r")
    ax[1].plot(range(1, num_samples + 1), last_predictions_z, label="Predicted Z", color="b")

    if show_real_values:
        ax[1].plot(range(1, num_samples + 1), last_targets_x, label="Target X", color="g")
        ax[1].plot(range(1, num_samples + 1), last_targets_y, label="Target Y", color="orange")
        ax[1].plot(range(1, num_samples + 1), last_targets_z, label="Target Z", color="purple")


    ax[1].legend(loc='best', title="Predicted vs Target (x, y, z)")
    ax[1].relim()
    ax[1].autoscale_view()

    # Show the final plot
    plt.ioff()  # Turn off interactive mode
    plt.show()

    torch.save(model.state_dict(), 'model_state_dict_lstm_smooth.pth')
    
    test(model, val_dataset, criterion, device,scaler=sclr,num_samples=500)
    test(model, val_dataset, criterion, device,scaler=sclr,num_samples=250)
    test(model, val_dataset, criterion, device,scaler=sclr,num_samples=100)
    test(model, val_dataset, criterion, device,scaler=sclr,num_samples=50)
    
    
    return model
    
    


def test(
    model, 
    val_dataset, 
    criterion, 
    device, 
    scaler=None, 
    batch_size=256, 
    num_samples=100
):
    model.eval()
    
    # Initialize variables to store predictions and targets
    all_predictions = []
    all_targets = []
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            outputs, _, _ = model(batch_X)
            
            all_predictions.append(outputs.cpu().detach().numpy())
            all_targets.append(batch_Y.cpu().detach().numpy())
    
    # Flatten the predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Reverse normalization if scaler is provided
    if scaler is not None:
        # Add padding if necessary to match the number of features the scaler was fitted on
        num_features = scaler.n_features_in_
        
        # Handle predictions
        if all_predictions.shape[1] < num_features:
            padded_predictions = np.zeros((all_predictions.shape[0], num_features))
            padded_predictions[:, :all_predictions.shape[1]] = all_predictions
            all_predictions = scaler.inverse_transform(padded_predictions)[:, :all_predictions.shape[1]]
        
        # Handle targets
        if all_targets.shape[1] < num_features:
            padded_targets = np.zeros((all_targets.shape[0], num_features))
            padded_targets[:, :all_targets.shape[1]] = all_targets
            all_targets = scaler.inverse_transform(padded_targets)[:, :all_targets.shape[1]]
    
    # Get the last `num_samples` samples
    last_predictions = all_predictions[-num_samples:]
    last_targets = all_targets[-num_samples:]
    
    # Plot the predictions vs real values
    plot_predictions_vs_targets(last_predictions, last_targets, num_samples)

    

def plot_predictions_vs_targets(predictions, targets, num_samples,output_file="results.txt"):
    # Extract x, y, and z values for plotting
    predictions_x = predictions[:, 0]
    predictions_y = predictions[:, 1]
    predictions_z = predictions[:, 2]

    targets_x = targets[:, 0]
    targets_y = targets[:, 1]
    targets_z = targets[:, 2]
    
    # Calculate absolute differences
    abs_diff_x = np.abs(predictions_x - targets_x)
    abs_diff_y = np.abs(predictions_y - targets_y)
    abs_diff_z = np.abs(predictions_z - targets_z)

    # Create a figure for the grid layout
    plt.figure(figsize=(16, 12))

    # X Component
    plt.subplot(2, 3, 1)
    plt.plot(range(1, num_samples + 1), predictions_x, label='Predicted X', color='r')
    plt.plot(range(1, num_samples + 1), targets_x, label='Target X', color='g')
    plt.title('Predicted vs Real X')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(range(1, num_samples + 1), abs_diff_x, label='|Predicted - Target| X', color='b', linestyle='--')
    plt.title('Absolute Difference for X')
    plt.xlabel('Sample')
    plt.ylabel('Absolute Difference')
    plt.legend()
    plt.grid(True)

    # Y Component
    plt.subplot(2, 3, 2)
    plt.plot(range(1, num_samples + 1), predictions_y, label='Predicted Y', color='r')
    plt.plot(range(1, num_samples + 1), targets_y, label='Target Y', color='g')
    plt.title('Predicted vs Real Y')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(range(1, num_samples + 1), abs_diff_y, label='|Predicted - Target| Y', color='b', linestyle='--')
    plt.title('Absolute Difference for Y')
    plt.xlabel('Sample')
    plt.ylabel('Absolute Difference')
    plt.legend()
    plt.grid(True)

    # Z Component
    plt.subplot(2, 3, 3)
    plt.plot(range(1, num_samples + 1), predictions_z, label='Predicted Z', color='r')
    plt.plot(range(1, num_samples + 1), targets_z, label='Target Z', color='g')
    plt.title('Predicted vs Real Z')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 6)
    plt.plot(range(1, num_samples + 1), abs_diff_z, label='|Predicted - Target| Z', color='b', linestyle='--')
    plt.title('Absolute Difference for Z')
    plt.xlabel('Sample')
    plt.ylabel('Absolute Difference')
    plt.legend()
    plt.grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()

      # Calculate average absolute differences
    avg_abs_diff_x = np.mean(abs_diff_x)
    avg_abs_diff_y = np.mean(abs_diff_y)
    avg_abs_diff_z = np.mean(abs_diff_z)

    with open(output_file, "a") as f:
        f.write(f"-------------------------------------\n")
        f.write(f"Number of samples {num_samples}\n")
        f.write(f"Average Absolute Difference for X: {avg_abs_diff_x:.4f}\n")
        f.write(f"Average Absolute Difference for Y: {avg_abs_diff_y:.4f}\n")
        f.write(f"Average Absolute Difference for Z: {avg_abs_diff_z:.4f}\n")
        f.write(f"-------------------------------------\n")
        f.write("\n")



def prepare_dataset():
    print("starting dataset preparation...")
    
    
    
    in_position_data = pd.read_csv("/home/pavle/op-ml/LSTM/lstmData/positionData.csv")
    in_exo_data=pd.read_csv("/home/pavle/op-ml/LSTM/lstmData/mergedExogenous.csv")

    timeSeriesPositionData = transform_to_timestamp(in_position_data)
    
    

    merged_data=pd.merge(timeSeriesPositionData, in_exo_data, on='Timestamp')
    
    train, test = train_test_split(merged_data, test_size=0.1,shuffle=False)
    
    
    
    return train,test
    
    

if __name__=="__main__":
    
    train_data,test_data=prepare_dataset()
    
    print(train_data)
    print(test_data)
    
    
    # Initialize model, loss, and optimizer
    
    trainF=False
    if trainF:
        model=train(train_data,num_epochs=7000) #maybe 6000 is a sweet spot
    
    else:
        seq_length = 175
        
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        columns_to_use = [
        'x', 'y', 'z', 'Field_Magnitude_nT', 'Speed_km_per_s', 'Proton_Density_n_per_cc',
        'Proton_Temperature_K', 'Flow_Pressure_nPa', 'Alfven_Mach_Number', 'Kp_index',
        'Lyman_alpha','total_mass_density',
        'pertrubation_r', 'pertrubation_theta', 'pertrubation_phi'
        ]
        
        
        data = train_data[columns_to_use].values
    
    
        time_index = np.arange(0, len(data))  # Create time index
        period = 0.073433 * 24 * 60  # Convert period to minutes (0.073433 days in minutes)
        sine_time = np.sin(2 * np.pi * time_index / period)  # Sine transformation
        cosine_time = np.cos(2 * np.pi * time_index / period)  # Cosine transformation
        
        # Add time features to the data
        data_with_time_features = np.column_stack((data, sine_time, cosine_time))
        
        
        X, y, sclr = create_sequences_multivariate(data_with_time_features, seq_length)
    
    

        # Convert data to PyTorch tensors and move to GPU if available
        testX = torch.tensor(X, dtype=torch.float32).to(device)
        testY = torch.tensor(y, dtype=torch.float32).to(device)
            
        dataset = TensorDataset(testX, testY)
        
        
        model = LSTMModel(input_dim=17, hidden_dim=70, layer_dim=3, output_dim=3, dropout_prob=0.2).to(device)
        criterion = nn.MSELoss()
        model.load_state_dict(torch.load("/home/pavle/op-ml/model_state_dict_lstm_smooth_best.pth"))
        test(model, dataset, criterion, device,scaler=sclr,num_samples=50)
        
    
    