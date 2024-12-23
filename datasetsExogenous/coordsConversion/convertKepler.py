import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator  # Hermite interpolation PCHIP
from datetime import datetime

# Define the gravitational parameter (mu) for the central body (e.g., Earth)
mu = 398600.4418  # Gravitational parameter (in km^3/s^2)

def convert_datetime_to_seconds(date_time_str):
    """
    Converts a date-time string in the format 'YYYY-MM-DD HH:MM:SS' to total seconds 
    since the Unix epoch.
    """
    # Convert the string to a datetime object
    dt = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
    
    # Return the total seconds from the Unix epoch
    return (dt - datetime(1970, 1, 1)).total_seconds()

def seconds_to_datetime(seconds):
    """
    Converts seconds since the epoch to a datetime string in 'YYYY-MM-DD HH:MM:SS' format.
    """
    epoch = datetime(1970, 1, 1)
    dt = epoch + pd.to_timedelta(seconds, unit='s')
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def cartesian_to_kepler(r, times, t_current):
    """
    Converts Cartesian coordinates (position and velocity) into Keplerian elements using Hermite interpolation.

    r: List of previous position vectors (each of shape (3,))
    times: List of previous times corresponding to positions r (in seconds)
    t_current: Current time at which to estimate the velocity and calculate Keplerian elements (in seconds)
    
    Returns: A dictionary of Keplerian elements
    """
    # Ensure the positions and times are numpy arrays
    r = np.array(r)
    times = np.array(times)
    
    # Create Hermite interpolators for position and velocity
    interpolator_position = PchipInterpolator(times, r, axis=0)
    
    # Calculate velocity by differentiating the position interpolator
    interpolator_velocity = interpolator_position.derivative(1)
    
    # Get the position and velocity at the current time
    r_current = interpolator_position(t_current)
    v_current = interpolator_velocity(t_current)
    
    # Calculate the magnitude of the position and velocity vectors
    r_mag = np.linalg.norm(r_current)
    v_mag = np.linalg.norm(v_current)
    
    # Calculate the specific angular momentum (h) = r x v
    h = np.cross(r_current, v_current)
    h_mag = np.linalg.norm(h)
    
    # Compute the eccentricity vector (e)
    e_vec = (np.cross(v_current, h) / mu) - (r_current / r_mag)
    e = np.linalg.norm(e_vec)  # Eccentricity
    
    # Semi-major axis (a) using the specific orbital energy
    energy = (v_mag**2 / 2) - (mu / r_mag)
    a = -mu / (2 * energy)  # Semi-major axis
    
    # Inclination (i) using the angular momentum vector (h)
    i = np.arccos(h[2] / h_mag)
    
    # Longitude of the ascending node (Ω)
    n = np.array([-h[1], h[0], 0])  # Node vector
    n_mag = np.linalg.norm(n)
    if n_mag != 0:
        Omega = np.arccos(n[0] / n_mag)
        if n[1] < 0:
            Omega = 2 * np.pi - Omega
    else:
        Omega = 0.0  # If n is a zero vector (in case of polar orbit)
    
    # Argument of perihelion (ω)
    if e != 0:
        omega = np.arccos(np.dot(n, e_vec) / (n_mag * e))
        if e_vec[2] < 0:
            omega = 2 * np.pi - omega
    else:
        omega = 0.0
    
    # True anomaly (ν)
    if e != 0:
        nu = np.arccos(np.dot(e_vec, r_current) / (e * r_mag))
        if np.dot(r_current, v_current) < 0:
            nu = 2 * np.pi - nu
    else:
        nu = 0.0
    
    # Return the orbital elements as a dictionary
    return {
        'semi_major_axis_a': a,
        'eccentricity_e': e,
        'inclination_i': np.degrees(i),  # Convert to degrees
        'longitude_of_ascending_node_Omega': np.degrees(Omega),  # Convert to degrees
        'argument_of_perihelion_omega': np.degrees(omega),  # Convert to degrees
        'true_anomaly_nu': np.degrees(nu),  # Convert to degrees
    }

def process_satellite_data(input_csv, output_csv, n_taps=20):
    # Step 1: Read the data from the CSV into a DataFrame
    data = pd.read_csv(input_csv)
    
    # Combine date and time columns, then convert to total seconds
    data['time'] = data['date'] + ' ' + data['time']  # Combine date and time
    data['time'] = data['time'].apply(convert_datetime_to_seconds)
    
    times = data['time'].values
    positions = data[['x', 'y', 'z']].values
    
    # Step 2: Use Hermite interpolation to estimate velocity and convert to Keplerian elements
    keplerian_elements_list = []
    original_times_list = []
    
    for i in range(n_taps, len(positions)):
        # Get the previous n_taps positions and corresponding times
        r = positions[i-n_taps:i]
        t_current = times[i]
        times_current = times[i-n_taps:i]
        
        # Get the corresponding Keplerian elements for the current position
        keplerian_elements = cartesian_to_kepler(r, times_current, t_current)
        
        # Append the results to the list
        keplerian_elements_list.append(keplerian_elements)
        
        # Save the original timestamp for the current point
        original_times_list.append(data['time'][i])
    
    # Step 3: Convert the list of Keplerian elements into a DataFrame
    keplerian_df = pd.DataFrame(keplerian_elements_list)
    
    # Convert original times (in seconds) back to datetime in the format 'YYYY-MM-DD HH:MM:SS'
    keplerian_df['timestamp'] = pd.Series(original_times_list).apply(seconds_to_datetime)
    
    # Step 4: Reorder the DataFrame so that the 'time' column is the first column
    keplerian_df = keplerian_df[['timestamp'] + [col for col in keplerian_df.columns if col != 'timestamp']]
    
    # Step 5: Save the Keplerian elements to a new CSV
    keplerian_df.to_csv(output_csv, index=False)
    print(f"Keplerian elements saved to {output_csv}")

# Example usage:
process_satellite_data('/home/pavle/op-ml/ProphetFnnOD/datasets/output.csv', 'keplerian_output.csv', n_taps=20)
