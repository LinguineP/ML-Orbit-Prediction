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
    """
    r = np.array(r)
    times = np.array(times)
    
    interpolator_position = PchipInterpolator(times, r, axis=0)
    interpolator_velocity = interpolator_position.derivative(1)
    
    r_current = interpolator_position(t_current)
    v_current = interpolator_velocity(t_current)
    
    r_mag = np.linalg.norm(r_current)
    v_mag = np.linalg.norm(v_current)
    
    h = np.cross(r_current, v_current)
    h_mag = np.linalg.norm(h)
    
    e_vec = (np.cross(v_current, h) / mu) - (r_current / r_mag)
    e = np.linalg.norm(e_vec)
    
    energy = (v_mag**2 / 2) - (mu / r_mag)
    a = -mu / (2 * energy)
    
    i = np.arccos(h[2] / h_mag)
    
    n = np.array([-h[1], h[0], 0])
    n_mag = np.linalg.norm(n)
    if n_mag != 0:
        Omega = np.arccos(n[0] / n_mag)
        if n[1] < 0:
            Omega = 2 * np.pi - Omega
    else:
        Omega = 0.0
    
    if e != 0:
        omega = np.arccos(np.dot(n, e_vec) / (n_mag * e))
        if e_vec[2] < 0:
            omega = 2 * np.pi - omega
    else:
        omega = 0.0
    
    if e != 0:
        nu = np.arccos(np.dot(e_vec, r_current) / (e * r_mag))
        if np.dot(r_current, v_current) < 0:
            nu = 2 * np.pi - nu
    else:
        nu = 0.0
    
    keplerian_elements = {
        'semi_major_axis_a': a,
        'eccentricity_e': e,
        'inclination_i': np.degrees(i),
        'longitude_of_ascending_node_Omega': np.degrees(Omega),
        'argument_of_perihelion_omega': np.degrees(omega),
        'true_anomaly_nu': np.degrees(nu),
    }
    
    # Convert to modified equanoctial coordinates
    equanoctial_elements = keplerian_to_equanoctial(keplerian_elements)
    
    return equanoctial_elements

def keplerian_to_equanoctial(keplerian_elements):
    """
    Converts Keplerian elements to modified equanoctial coordinates.
    """
    a = keplerian_elements['semi_major_axis_a']
    e = keplerian_elements['eccentricity_e']
    i = np.radians(keplerian_elements['inclination_i'])
    Omega = np.radians(keplerian_elements['longitude_of_ascending_node_Omega'])
    omega = np.radians(keplerian_elements['argument_of_perihelion_omega'])
    nu = np.radians(keplerian_elements['true_anomaly_nu'])
    
    # Calculate the modified equanoctial coordinates
    p = a * (1 - e**2)
    f = np.cos(omega) * np.sin(i)
    g = np.sin(omega) * np.sin(i)
    h = np.cos(i) * np.cos(Omega)
    k = np.cos(i) * np.sin(Omega)
    l = nu + omega - Omega
    
    return {
        'p': p,
        'f': f,
        'g': g,
        'h': h,
        'k': k,
        'l': np.degrees(l),  # Convert l to degrees
    }

def process_satellite_data(input_csv, output_csv, n_taps=20):
    data = pd.read_csv(input_csv)
    data['time'] = data['date'] + ' ' + data['time']
    data['time'] = data['time'].apply(convert_datetime_to_seconds)
    
    times = data['time'].values
    positions = data[['x', 'y', 'z']].values
    
    equanoctial_elements_list = []
    original_times_list = []
    
    for i in range(n_taps, len(positions)):
        r = positions[i-n_taps:i]
        t_current = times[i]
        times_current = times[i-n_taps:i]
        
        equanoctial_elements = cartesian_to_kepler(r, times_current, t_current)
        
        equanoctial_elements_list.append(equanoctial_elements)
        original_times_list.append(data['time'][i])
    
    equanoctial_df = pd.DataFrame(equanoctial_elements_list)
    
    equanoctial_df['timestamp'] = pd.Series(original_times_list).apply(seconds_to_datetime)
    
    equanoctial_df = equanoctial_df[['timestamp'] + [col for col in equanoctial_df.columns if col != 'timestamp']]
    
    equanoctial_df.to_csv(output_csv, index=False)
    print(f"Modified equanoctial elements saved to {output_csv}")

# Example usage:
process_satellite_data('/home/pavle/op-ml/ProphetFnnOD/datasets/output.csv', 'equanoctial_output.csv', n_taps=20)
