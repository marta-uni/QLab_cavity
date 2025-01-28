import os
import pandas as pd
import matplotlib.pyplot as plt
import functions as fn


'''This code reads the csv files, crops them to a window where the piezo voltage is just increasing,
removes NaNs and fits the piezo data linearly. It then saves a csv file with  4 columns: timestamps,
transmission, volt_piezo, piezo_fitted.'''

#####################################################################################################
# section with file specific values

# Define the folder and file paths
folder_name = 'data_bessel'
title = 'bessel00021'
df = pd.DataFrame()
# Import single channel files
channel_files = [f'{folder_name}/raw/C{i}{title}.csv' for i in range(1, 5)]
# Define channel names
column_titles = ['transmission', 'reflection', 'volt_piezo', 'total_intensity']
os.makedirs(f"{folder_name}/clean_data", exist_ok=True)
os.makedirs(f"{folder_name}/figures/simple_plots_time", exist_ok=True)
# Boolean to activate extra cropping, if necessary:
cropping = False
# You have to set the mask variable to what you need in the code
# If you are using less than 4 channels you have to modify the rest of the code

#####################################################################################################

# Loop through each file in the file_paths
for i, channel in enumerate(channel_files):
    file_name = os.path.basename(channel).replace(
        '.csv', '')  # Extract file name without extension

    # Read the CSV file, skip the first 5 rows, and specify the data types
    # Encoding is different from usual utf-8, so we need to specify it.
    data = pd.read_csv(channel, skiprows=5, names=['timestamp', 'ch'], encoding='cp1252', dtype={
                       'timestamp': float, 'ch': float})

    # Write timestamp column just once
    if df.empty:
        df['timestamp'] = data.iloc[:, 0]

    # Write channel reading to dataframe
    df[column_titles[i]] = data.iloc[:, 1]


df = df.dropna()

# Producing single numpy arrays for manipulation with functions
# This may not be the most efficient way to handle this but it works at least
timestamps = df['timestamp'].to_numpy()
transmission = df['transmission'].to_numpy()
reflection = df['reflection'].to_numpy()
volt_piezo = df['volt_piezo'].to_numpy()
total_intensity = df['total_intensity'].to_numpy()

# Extra cropping, use it in case of mode hopping
if cropping:
    mask = (timestamps >= -0.006)
    timestamps = timestamps[mask]
    transmission = transmission[mask]
    reflection = reflection[mask]
    volt_piezo = volt_piezo[mask]
    total_intensity = total_intensity[mask]

# Cropping data to a single sweep
# NOTICE, VOLT PIEZO HAS TO BE THE FIRST ONE
volt_piezo, timestamps, transmission, reflection, total_intensity = fn.crop_to_min_max(
    volt_piezo, timestamps, transmission, reflection, total_intensity)

piezo_fitted = fn.fit_piezo_line(timestamps, volt_piezo)

# Plotting, just for fun
plt.figure()
plt.scatter(timestamps, transmission, label='Transmission',
         color='blue', s=5, marker='.')
plt.scatter(timestamps, piezo_fitted, label='Piezo voltage',
         color='red', s=5, marker='.')
plt.scatter(timestamps, total_intensity, label='Total intensity',
         color='green', s=5, marker='.')
plt.scatter(timestamps, reflection, label='Reflection',
         color='purple', s=5, marker='.')
plt.xlabel('Timestamp [s]')
plt.ylabel('Channel Value [V]')
plt.title('Timestamp vs Channel Data')
plt.grid()
plt.legend()
plt.savefig(f'{folder_name}/figures/simple_plots_time/{title}.png')
plt.show()

# Saving data in clean_data folder
output_file = f'{folder_name}/clean_data/{title}.csv'
df = df.iloc[:len(timestamps)].reset_index(drop=True)
df['timestamp'] = timestamps
df['transmission'] = transmission
df['reflection'] = reflection
df['volt_piezo'] = piezo_fitted
df['total_intensity'] = total_intensity

df.to_csv(output_file, index=False)
print(f"Data saved to {output_file}")
