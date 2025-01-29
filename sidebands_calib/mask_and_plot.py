import pandas as pd
import matplotlib.pyplot as plt
import os

#THIS FILE ZOOMS IN AND PLOTS THE VOLT_PIEZO VS TRANSMISSION

#list of file titles
title_list = [f'error{str(i).zfill(5)}' for i in range(1, 19)]
titles_list = title_list[:4] + title_list[5:7] + title_list[10:18]
##########################################################################
#option to mask

frequencies = [50, 40, 30, 20, 10, 5, 40, 40, 40, 40, 40, 40, 40, 40] #MHz
lower_bounds = [1.47, 1.35, 1.19, 1.10, 0.75, 0.478, 1.07, 0.97, 0.94, 0.90, 0.90, 0.90, 0.89, 0.85]
upper_bounds = [1.58, 1.43, 1.25, 1.15, 0.82, 0.527, 1.17, 1.07, 1.05, 1.03, 1.03, 1.00, 0.98, 0.95]

###########################################################################

# Directory to save the plots
output_dir = 'peaks_errorfiles/figures/mask'
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
filtered_data_dir = 'clean_data'
os.makedirs(filtered_data_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Loop through the list of titles
for i, title in enumerate(titles_list):
    file_path = f"data/{title}.csv"  # Assuming the files have a .csv extension
    try:
        # Load the data and save into new file
        data = pd.read_csv(file_path)
        data = data[(data['volt_piezo'] > lower_bounds[i]) & (data['volt_piezo'] < upper_bounds[i])]
        output_file = f"{filtered_data_dir}/{title}_filtered.csv"
        data.to_csv(output_file, index=False)  # Save without the index column

        # Plot transmission vs. volt_piezo
        plt.figure(figsize=(8, 6))
        plt.scatter(data['volt_piezo'], data['transmission'], marker='.', linestyle='-', color='b', label='Transmission')
        plt.xlabel('Volt Piezo')
        plt.ylabel('Transmission')
        plt.yscale('log')
        plt.ylim(1e-3, None)  # Lower limit at 10^-3; no upper limit specified
        plt.title(f'Transmission vs Volt Piezo - {title}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save the plot as a PDF
        output_file = os.path.join(output_dir, f"{title}_plot.png")
        plt.savefig(output_file, format='png')
        plt.close()  # Close the figure to free memory

        print(f"Plot saved: {output_file}")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")


