import os
import pandas as pd
import functions as fn
import matplotlib.pyplot as plt


'''This code reads the csv files, crops them to a window where the piezo voltage is just increasing,
removes NaNs and fits the piezo data linearly. It then saves a csv file with  4 columns: timestamps,
volt_laser, volt_piezo, piezo_fitted.'''

# Define the folder and file paths
folder_name = 'data_confocal'
file_paths = [f"{folder_name}/raw/scope_{i}.csv" for i in range(15, 26)]
os.makedirs(f"{folder_name}/clean_data", exist_ok=True)
os.makedirs(f"{folder_name}/figures/simple_plots_time", exist_ok=True)

# Loop through each file in the file_paths
for file_path in file_paths:
    file_name = os.path.basename(file_path).replace(
        '.csv', '')  # Extract file name without extension

    # Read the CSV file, skip the first 2 rows, and specify the data types
    data = pd.read_csv(file_path, skiprows=2, names=['timestamp', 'volt_laser', 'volt_piezo'], dtype={
                       'timestamp': float, 'volt_laser': float, 'volt_piezo': float})

    # Remove any rows with NaN values
    data_cleaned = data.dropna()

    # Extract the columns
    timestamps = data_cleaned['timestamp'].to_numpy()
    volt_laser = data_cleaned['volt_laser'].to_numpy()
    volt_piezo = data_cleaned['volt_piezo'].to_numpy()

    # Crop data to one piezo cycle
    volt_piezo, timestamps, volt_laser = fn.crop_to_min_max(volt_piezo, timestamps, volt_laser)

    # Fit the piezo data
    piezo_fitted = fn.fit_piezo_line(timestamps, volt_piezo)
    
    # Plotting, just for fun
    plt.figure()
    plt.plot(timestamps, volt_laser, label='Laser intensity',
             color='blue', markersize=5, marker='.')
    plt.plot(timestamps, piezo_fitted/10, label='Piezo voltage/10',
             color='red', markersize=5, marker='.')
    plt.xlabel('Timestamp [s]')
    plt.ylabel('Channel Value [V]')
    plt.title('Timestamp vs Channel Data')
    plt.grid()
    plt.legend()
    plt.savefig(f'{folder_name}/figures/simple_plots_time/{file_name}.png')
    plt.close()

    # Adjust the file path as needed
    output_file = figure_name = f'{folder_name}/clean_data/{file_name}.csv'
    data = {
        'timestamp': timestamps,
        'volt_laser': volt_laser,
        'volt_piezo': volt_piezo,
        'piezo_fitted': piezo_fitted
    }
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
