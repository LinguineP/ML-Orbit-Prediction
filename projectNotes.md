## CDDIS_larets y2022 (1 1 2022- 1 1 2023)

### data format not correct

`.dgf (Data Generation File)`
Purpose: These files contain detailed data about the satellite’s orbit, which is essential for generating predictions and analyses.
Content: They often include parameters necessary for precise orbit determination, such as satellite position and velocity vectors, and other orbital elements.
Usage: Used by researchers and analysts to create accurate models of satellite trajectories.
`.hts (High-Precision Time Series)`
Purpose: HTS files provide high-precision time series data, which is crucial for tracking the satellite’s position over time.
Content: These files typically include time-stamped position data, allowing for high temporal resolution in tracking.
Usage: Essential for applications that require precise timing and positioning, such as geodesy and Earth observation.
`.sgf (Standard Geodetic Format)`
Purpose: SGF files store geodetic data in a standardized format, ensuring compatibility across different geodetic software and systems.
Content: They include various geodetic measurements and parameters, formatted according to international standards.
Usage: Used widely in the geodetic community for data exchange and analysis.
`.mcc (Mission Control Center)`
Purpose: MCC files are used by mission control centers for operational purposes.
Content: These files may contain commands, telemetry data, or other mission-specific information necessary for satellite operations.
Usage: Critical for the day-to-day operations and management of satellite missions.
Additional Resources
For more detailed information, you can refer to the following resources:

ILRS Missions1 https://ilrs.cddis.eosdis.nasa.gov/missions/satellite_missions/current_missions/lare_general.html
CDDIS Data and Derived Products2 https://cddis.nasa.gov/Data_and_Derived_Products/SLR/Precise_orbits.html
These documents provide in-depth insights into the technology and applications of satellite laser ranging, including the use of various data file formats.

## Francisco mail

Regarding file formats, these indicate the data provider:

- .dgf: Deutsches Geodätisches Forschungsinstitut (TUM)
- .sgf: Space Geodesy Facility (UK)
- etc

I used the **hts** format because it originates from NASA, and the Americans have a broader network of ground stations more evenly distributed around the planet. I do recommend using the `slrfield` package for downloading the data, It mostly works, but it might need some tweaks, I don't quite remember, but it did have a bug the first time I tried to download the data. Additionally, the Eurolas Data Center website was incredibly helpful: https://edc.dgfi.tum.de/en/.

After downloading the data, I used the `slrfield` function `CPF.from_data`, which extracts relevant information. I took the 'positions[m]' 'Time of Ephemeris Production'and 'ts_utc'. With this, I can match each `ts_utc` to the position issued closest to the time of ephemeris production. Usually, there’s one position per day, with only a few gaps to account for.

If you’re starting to use Prophet, here are a few tips from my experience, because it took some time to make it work effectively.Finding the frequencies (and the number of frequencies) for each time-series will have a very significant impact on the quality of the model, also, there is quite a bit of fine-tuning involved, I used optuna to go through every single parameter choice.
