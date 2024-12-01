"""
The format for each word is as contained in the ftp-accessible annual ASCII files. Each record contains 56 words as described below.

WORD TYPE Fill values    MEANING                UNITS/COMMENTS

 1    I4                  Year                   1963,1964,1965...
 2    I4                  Decimal Day            Day of year (Jan 1 = Day 1)
 3    I3                  Decimal Hour           (0, 1, ...23; average for "1" is from
                                                              01:00 to 02:00)
 4   I5       9999       Bartels Rotation Number
 5   I3         99       ID for IMF SC               See table below
 6   I3         99       ID for SW Plasma SC         See table below
 7   I4        999      # of fine time scale
                          points in IMF Avgs
 8  I4        999       # of fine time scale
                          points in Plasma Avgs
 9  F6.1    999.9       Field Magnitude Avg,      (scalar)
                          <F>                  nT
10  F6.1    999.9       Magnitude of Average        nT
                          Field vector, |<B>|
11  F6.1    999.9        Lat. Angle of avg.          Deg (GSE Coords)
                                   Field vector
12  F6.1    999.9        Long. Angle of avg.         Deg (GSE Coords)
                                  Field vector
13  F6.1    999.9        Bx,GSE                     (nT)
14  F6.1    999.9        By,GSE                     (nT)
15  F6.1    999.9        Bz,GSE                     (nT)
16  F6.1    999.9        By,GSM                     (nT) see footnote 4
17  F6.1    999.9        Bz,GSM                     (nT) see footnote 4
18  F6.1    999.9        sigma-|B|                   RMS Standard deviation in avg
                                                      magnitude (wd. 10), nT
19  F6.1    999.9        sigma-B                     RMS Standard deviation in field
                                                     vector, in nT; see footnote 3
20  F6.1    999.9        sigma-Bx                    RMS Standard deviation in GSE X
                                                     comp. av, nT
21  F6.1    999.9        sigma-By                    RMS Standard deviation in GSE Y
                                                     comp, av, nT
22  F6.1    999.9        sigma-Bz                    RMS standard deviation in GSE Z
                                                     comp, av, nT
23  F9.0 9999999.       Proton temperature      Degrees Kelvin
24  F6.1    999.9        Proton density              #N/cm**3
25  F6.0    9999.        Bulk speed                  Km/sec. (scalar)
26  F6.1    999.9        Bulk flow longitude        phi-V, degrees. phi-V increases positively/
                                                   negatively  from zero as the flow direction
                                                   changes from being along
                                                    the -Xgse axis toward the +Ygse/-Ygse axis;
                                                   see footnote 1)
  
27  F6.1    999.9        Bulk flow latitude     theta-V, degrees. theta-V increases positively/
                                                negatively from zero as the flow direction changes
                                                from being in 
                                                 the Xgse-Ygse plane toward the +Zgse/-Zgse axis;
                                                see footnote 1)

28  F6.3    9.999        Na/Np               Alpha/Proton ratio 
29  F6.2    99.99       Flow Pressure       P (nPa) = (1.67/10**6) * Np*V**2 * (1+ 4*Na/Np)
                                           for hours with non-fill Na/Np ratios and
                                            P (nPa) = (2.0/10**6) * Np*V**2
                                            for hours with fill values for Na/Np 
                                            For details click HERE 
   
30  F9.0  9999999.      sigma-T   Degrees Kelvin
31  F6.1    999.9       sigma-n     cm -3
32  F6.0    9999.       sigma-V     km/sec
33  F6.1    999.9       sigma-phi-V   deg
34  F6.1    999.9       sigma-theta-V  deg
35  F6.3    9.999       sigma-ratio   

36  F7.2   999.99       Electric field         mV/m, -V(km/s) * Bz (nT; GSM) * 10**-3 
37  F7.2   999.99       Plasma beta            Beta = [(T*4.16/10**5) + 5.34] * Np / B**2, 
                                                    For details click HERE 
38  F6.1    999.9        Alfven mach number     Ma = (V * Np**0.5) /(20 * B) , 
                                                For details click HERE  

39  I3         99       Kp*10                  3-hr Kp index from  GFZ, Potsdam
                                                        (See footnote 2)
40  I4        999       R                      Daily New sunspot Number (version 2) from 
                                                         http://sidc.oma.be/silso/datafiles
                                                        Details HERE

41  I6      99999       DST Index              nT, from Kyoto
42  I5       9999       AE-index               nT, from Kyoto

43 F10.2 999999.99     PROT Flux               1/(cm^2 sec ster), >1 MeV
44 F9.2  99999.99       PROT Flux              1/(cm^2 sec ster), >2 MeV
45 F9.2  99999.99      PROT Flux               1/(cm^2 sec ster), >4 MeV
46 F9.2  99999.99      PROT Flux               1/(cm^2 sec ster), >10 MeV
47 F9.2  99999.99      PROT Flux               1/(cm^2 sec ster), >30 MeV
48 F9.2  99999.99       PROT Flux              1/(cm^2 sec ster), >60 MeV
49  I3    0               M'SPH Flux Flag        = 6: No m'spheric contribution
                                                 = 5: M'sph contrib in lowest
                                                       energy channel
                                                 = 4: M'sph c in lowest 2 chnls
                                                          ...
                                                  = 1: M'sph c in lowest 5 chnls
                                                  = 0: M'sph c in all channels
                                                  =-1: Not checked for M'sph c:
                                                      relevant after 88/306
                                                     (See Flux)

50  I4    999     ap-index              3-hr ap index, nT, from GFZ, Potsdam
51  F6.1 999.9     f10.7_index           Daily index,(10**-22) Watts/meter sq/hertz
                                               adjusted to 1AU from Canada
52  F6.1 999.9       PC(N)               DTU,Data Center for Geomagnetism, Copenhagen 
53  I6   99999       AL-index             nT, from Kyoto                     
54  I6   99999       AU-index             nT, from Kyoto
55  F5.1  99.9       MAC            Magnetosonic mach number  =V/Magnetosonic_speed 
                                     Magnetosonic speed = [(sound speed)**2 + 
                                     (Alfv speed)**2]**0.5
                                     The Alfven speed = 20. * B / N**0.5 
                                    The sound speed = 0.12 * [T + 1.28*10**5]**0.5 
                                    For details click HERE  

56 F9.6 0.999999                     Daily Solar Lyman-alpha, W/m^2   
                                     See http://lasp.colorado.edu/lisird/lya/

57 F7.4  9.9999                     Proton Quasy-Invariant (QI)=(B^2/8*pi)/(Den*V^2/2)
                                    QI ==SW (magnetic energy density)/(kinetic energy density)
                                    For details click HERE



--------------------------------------------------------------------------
FORMAT(2I4,I3,I5,2I3,2I4,14F6.1,F9.0,F6.1,F6.0,2F6.1,F6.3,F6.2,
F9.0,F6.1,F6.0,2F6.1,F6.3,2F7.2,F6.1,I3,I4,I6,I5,F10.2,5F9.2,I3,I4,2F6.1,2I6,F5.1,F9.6,F7.4)
Note that for missing data, fill values consisting of a blank followed by 9's which together constitute the Ix or Fx.y format are used.
Presengages of coverage of real mag and plasma data for each year are shown HERE
"""

import pandas as pd
import numpy as np

# Read the file with a flexible separator (tabs or spaces)
df = pd.read_csv(
    ".\\omnidata\\downloaded_files\\omni_min_vh1min.lst",
    delim_whitespace=True,  # Handles both tabs and multiple spaces
    header=None,  # No header in the file
)

# Optionally assign column names for clarity
df.columns = [
    "Year",
    "Day",
    "Hour",
    "Minute",
    "Timeshift",
    "Field_Magnitude_nT",
    "Speed_km_per_s",
    "Proton_Density_n_per_cc",
    "Proton_Temperature_K",
    "Flow_Pressure_nPa",
    "Alfven_Mach_Number",
]

original_df = df.copy()

# Example of placeholder values to replace
placeholder_values = [99999.9, 999.99, 9999999.0, 99.99, 999.9]

# Replace placeholder values with NaN
df.replace(placeholder_values, np.nan, inplace=True)

# Interpolate missing values (linear interpolation by default)
df.interpolate(method="linear", inplace=True)

df.bfill(axis=0, inplace=True)  # solves the edge case where first value in a col is NaN

df.ffill(axis=0, inplace=True)  # solves the edge case where last value in a col is NaN


if False:  # change true/false based on weather you want the plot or not
    # plots dont show much because scale of fill values is to great compared to original vals
    # Loop through each column and plot
    import matplotlib.pyplot as plt

    columns_to_plot = [
        "Speed_km_per_s",
        "Proton_Density_n_per_cc",
        "Proton_Temperature_K",
        "Flow_Pressure_nPa",
        "Field_Magnitude_nT",
    ]

    for col in columns_to_plot:
        plt.figure(figsize=(12, 6))
        plt.plot(
            original_df.iloc[:100].index,
            original_df.iloc[:100][col],
            label=f"Original {col} ",
            linestyle="dotted",
            color="gray",
        )
        plt.plot(
            df.iloc[:100].index,
            df.iloc[:100][col],
            label=f"new vals {col}",
            linestyle="solid",
        )
        plt.xlabel("Row Index")
        plt.ylabel(col)
        plt.title(f"Comparison of Interpolation Methods for {col}")
        plt.legend()
        plt.show()


# turns givent time data into
df["Timestamp"] = (
    pd.to_datetime(df["Year"] * 1000 + df["Day"], format="%Y%j")
    + pd.to_timedelta(df["Hour"], unit="h")
    + pd.to_timedelta(df["Minute"], unit="m")
)


df = df.drop(columns=["Year", "Day", "Hour", "Minute", "Timeshift"])


df = df[["Timestamp"] + [col for col in df.columns if col != "Timestamp"]]

df.to_csv("D:\\fax\\master\\op-ml\\omnidata\\cleanedData\\cleaned1min.csv", index=False)


print(df)
