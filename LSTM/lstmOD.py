import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
PLOT_FLAG=True




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE=256
MODEL_PATH="/home/pavle/op-ml/model_state_dict_lstm_smooth1.pth"


TEST_OUT_FILE="results.txt"


def transform_to_timestamp(in_data):
    df = pd.DataFrame(in_data)
    df["Timestamp"] = df["date"] + " " + df["time"]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S")
    df["Timestamp"] = df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df.drop(["date", "time"], axis=1)
    df = df[["Timestamp", "x", "y", "z"]]
    return df


def plot_and_output_predictions_vs_targets(predictions, targets, num_samples):
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
    if PLOT_FLAG:
    
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

    with open(TEST_OUT_FILE, "a") as f:
        f.write(f"-------------------------------------\n")
        f.write(f"Number of samples {num_samples}\n")
        f.write(f"Average Absolute Difference for X: {avg_abs_diff_x:.4f}\n")
        f.write(f"Average Absolute Difference for Y: {avg_abs_diff_y:.4f}\n")
        f.write(f"Average Absolute Difference for Z: {avg_abs_diff_z:.4f}\n")
        f.write(f"-------------------------------------\n")
        f.write("\n")
        
    



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
        return xs, ys,None


def validate(validation_data, model, criterion, device):
    model.eval()  # Set the model to evaluation mode
    val_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False)

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


def train(dataset,num_epochs = 5):
    print("Training..")


    # Initialize model and move it to the GPU if available
    model = LSTMModel(input_dim=17, hidden_dim=70, layer_dim=3, output_dim=3, dropout_prob=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001) #optim


    # Training loop
    
    h0, c0 = None, None  # Initialize hidden and cell states
    
    plt.ion()  # Turn on interactive mode for real-time plotting
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))  # Create two subplots: one for loss and one for predicted vs expected

    # Loss plot
    loss_line, = ax.plot([], [], marker='o', color='b')  # Empty plot for loss initially
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Loss vs Epochs')
    ax.grid(True)


    losses = []  # To store loss values
    last_epoch_predictions = []  # To store predictions for the last epoch
    last_epoch_targets = []  # To store targets for the last epoch


    
    train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=False)
    
    for batch_X, batch_Y in train_loader:
        print(batch_X[0].shape)
        exit()

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
        ax.relim()  # Recalculate the limits of the plot
        ax.autoscale_view()  # Autoscale the view to fit new data

        if (epoch + 1) % 5 == 0:
            plt.pause(0.1)  # Pause to update the plot
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            validation_loss = validate(dataset["val"], model, criterion, device)
            print(f"Validation Loss: {validation_loss:.4f}")


    plt.ioff()  # Turn off interactive mode
    plt.show()
    
    torch.save(model.state_dict(), MODEL_PATH)
    
    return model
    
    


def test(dataset, num_samples=100):    
    
    print("Testing...")
    
    model = LSTMModel(input_dim=17, hidden_dim=70, layer_dim=3, output_dim=3, dropout_prob=0.2).to(device)
    criterion = nn.MSELoss()
    model.load_state_dict(torch.load(MODEL_PATH,weights_only=True))
    
    model.eval()
    
    # Initialize variables to store predictions and targets
    all_predictions = []
    all_targets = []
    
    test_loader = DataLoader(dataset["test"], batch_size=BATCH_SIZE, shuffle=False)
    scaler=dataset["scaler"]

    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            outputs, _, _ = model(batch_X)
            
            all_predictions.append(outputs.cpu().detach().numpy())
            all_targets.append(batch_Y.cpu().detach().numpy())
    
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Reverse normalization if scaler is provided
    if scaler is not None:
        # Add padding if necessary to match the number of features the scaler was fitted on
        num_features = scaler.n_features_in_
        
        # predictions
        if all_predictions.shape[1] < num_features:
            padded_predictions = np.zeros((all_predictions.shape[0], num_features))
            padded_predictions[:, :all_predictions.shape[1]] = all_predictions
            all_predictions = scaler.inverse_transform(padded_predictions)[:, :all_predictions.shape[1]]
        
        # targets
        if all_targets.shape[1] < num_features:
            padded_targets = np.zeros((all_targets.shape[0], num_features))
            padded_targets[:, :all_targets.shape[1]] = all_targets
            all_targets = scaler.inverse_transform(padded_targets)[:, :all_targets.shape[1]]
    
    # number of samples to take into account
    last_predictions = all_predictions[-num_samples:]
    last_targets = all_targets[-num_samples:]
    
    
    plot_and_output_predictions_vs_targets(last_predictions, last_targets, num_samples)

    




def prepare_dataset(train=0.8,validate=0.05,test=0.15,seq_length = 175):
    print("starting dataset preparation...")
    
    
    assert train+validate+test==1.0
    
    
    in_position_data = pd.read_csv("/home/pavle/op-ml/LSTM/lstmData/positionData.csv")
    in_exo_data=pd.read_csv("/home/pavle/op-ml/LSTM/lstmData/mergedExogenous.csv")

    timeSeriesPositionData = transform_to_timestamp(in_position_data)
    
    

    merged_data=pd.merge(timeSeriesPositionData, in_exo_data, on='Timestamp')
    
    
    
    columns_to_use = [
        'x', 'y', 'z', 'Field_Magnitude_nT', 'Speed_km_per_s', 'Proton_Density_n_per_cc',
        'Proton_Temperature_K', 'Flow_Pressure_nPa', 'Alfven_Mach_Number', 'Kp_index',
        'Lyman_alpha','total_mass_density',
        'pertrubation_r', 'pertrubation_theta', 'pertrubation_phi'
    ]
    
    
    
    data = merged_data[columns_to_use].values
    
    
    time_index = np.arange(0, len(data))  # Create time index
    period = 0.073433 * 24 * 60  # Convert period to minutes (0.073433 days in minutes)
    sine_time = np.sin(2 * np.pi * time_index / period)  # Sine transformation
    cosine_time = np.cos(2 * np.pi * time_index / period)  # Cosine transformation
    
    
    data_with_time_features = np.column_stack((data, sine_time, cosine_time))
    
    
    seq_length = 175 #175(5 days)
    X, y, sclr = create_sequences_multivariate(data_with_time_features, seq_length)
    
    # Convert data to PyTorch tensors and move to GPU if available
    dataX = torch.tensor(X, dtype=torch.float32).to(device)
    dataY = torch.tensor(y, dtype=torch.float32).to(device)
    
    dataset = TensorDataset(dataX, dataY)
    
    train_size = int(train * len(dataset))  
    validate_size = int(validate * len(dataset))  
    test_size = len(dataset) - train_size - validate_size  

    #Sequential split
    train_dataset = TensorDataset(dataX[:train_size], dataY[:train_size])
    val_dataset = TensorDataset(dataX[train_size:train_size + validate_size], dataY[train_size:train_size + validate_size])
    test_dataset = TensorDataset(dataX[train_size + validate_size:], dataY[train_size + validate_size:])

    # dataset dict
    dataset={"train":train_dataset,"val":val_dataset,"test":test_dataset,"seq_length":seq_length,"scaler":sclr}
    
    return dataset
    
    
    
def main():
    dataset=prepare_dataset()
    
    print(dataset)
    
    model=train(dataset,num_epochs=5) #maybe 6000 is a sweet spot
        
    test(dataset,num_samples=500)
    
    

if __name__=="__main__":
    
    main()