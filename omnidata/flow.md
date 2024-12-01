# Extracting data from OMNI Data

### _note all these work for my specific use case_

### FORMAT OF THE SUBSETTED FILE 1 minute

    ITEMS                      FORMAT

1 Year I4  
 2 Day I4  
 3 Hour I3  
 4 Minute I3  
 5 Timeshift I7  
 6 Field magnitude average, nT F8.2  
 7 Speed, km/s F8.1  
 8 Proton Density, n/cc F7.2  
 9 Proton Temperature, K F9.0  
10 Flow pressure, nPa F6.2  
11 Alfven mach number F6.1

### FORMAT OF THE SUBSETTED FILE 1 hour

    ITEMS                      FORMAT

1 YEAR I4  
 2 DOY I4  
 3 Hour I3  
 4 Kp index I3  
 5 Lyman_alpha F9.6

where these are the formats for 1 min and 1 hour

## 1) Download data

    output in downloaded_data

#### _downloadOmnidata1hour.py_

#### _downloadOmnidata1min.py_

## 2) Prepare 1 min and one hour

    output in cleanedData

#### _omnidata1min_clean.py_

    - this script solves any present fill values and reformats the input into a csv with format:
     Timestamp,Field_Magnitude_nT,Speed_km_per_s,Proton_Density_n_per_cc,Proton_Temperature_K,Flow_Pressure_nPa,Alfven_Mach_Number

     ~  fill_value_replacement_viz.py was used to view the effect of different fill strategies for 1min data ~

#### _omnidata1hour_clean.py_

    -this script streches the hourly data to reflect minutes  and outputs a csv in the following format:
    Timestamp,Kp_index,Lyman_alpha

## 3) Merge the 1hour and 1minute into complete omnidata

    output in omniComplete

#### _mergeExogeneous.py_

    merges the hourly and minute data
