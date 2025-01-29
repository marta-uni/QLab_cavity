import pandas as pd
import numpy as np
import os
import functions as fn
import matplotlib.pyplot as plt
from uncertainties import ufloat
import uncertainty_functions as ufn

################################LOAD DATA##############################
omegas = [50, 40, 30, 20, 10, 40, 40, 40, 40, 40, 40, 40] #MHz
print(len(omegas))
# Load the CSV file into a DataFrame
file_path = "fitted_parameters.csv"  # Replace with your file's path
data = pd.read_csv(file_path)

# Filter the data for files with exactly 3 peaks
files_with_three_peaks = data.groupby("file_name").filter(lambda x: len(x) == 3)
print(len(files_with_three_peaks))
# Extract parameters into arrays for files with 3 peaks
file_names = files_with_three_peaks["file_name"].unique()
peak_A = []
peak_x0 = []
peak_gamma = []
#peak_offset = []

#these arrays now include "subarrays" of triplets of peak parameters, stored as ufloats
for file in file_names:
    file_data = files_with_three_peaks[files_with_three_peaks["file_name"] == file]
    peak_A.append([ufloat(a, da) for a, da in zip(file_data["A"].values, file_data["A_uncertainty"].values)])
    peak_x0.append([ufloat(x0, dx0) for x0, dx0 in zip(file_data["x0"].values, file_data["x0_uncertainty"].values)])
    peak_gamma.append([ufloat(g, dg) for g, dg in zip(file_data["gamma"].values, file_data["gamma_uncertainty"].values)])
    #peak_offset.append([ufloat(o, do) for o, do in zip(file_data["offset"].values, file_data["offset_uncertainty"].values)])

# Convert lists to numpy arrays
peak_A = np.array(peak_A, dtype=object)
peak_x0 = np.array(peak_x0, dtype=object)
peak_gamma = np.array(peak_gamma, dtype=object)
#peak_offset = np.array(peak_offset, dtype=object)


# Main Calibration Loop
calibration_coefficients = []

for x0_list, omega, file in zip(peak_x0, omegas, file_names):
    expected_freq = np.arange(0, omega * 3, omega)

    # Perform the linear fit
    coeffs1, d_coeffs1 = ufn.plot_ufloat_fit(
        x0_list, expected_freq, ufn.lin_model,
        "Peaks in piezo voltage (V)", "Expected frequency (MHz)",
        f'Calibration of {file}', beta0=[1, 0],
        file_name=f'calibration/figures/{file}_calibration.png', save=True
    )

    # Save the calibration coefficients
    calibration_coefficients.append((file, coeffs1, d_coeffs1))

    # Load the data file
    data_file = pd.read_csv(f'clean_data/{file}_filtered.csv')
    volt_piezo = data_file['volt_piezo'].to_numpy()

    # Calibrate the volt_piezo values and add as a new column
    calibrated_frequencies = ufn.calibrate(volt_piezo, coeffs1)
    data_file['frequencies'] = calibrated_frequencies

    # Save the updated data file
    data_file.to_csv(f'clean_data/calibrated_{file}_filtered.csv', index=False)

# Save all calibration coefficients to a CSV file
calibration_coefficients = [
    (file_name, coeffs[0], d_coeffs[0], coeffs[1], d_coeffs[1])
    for file_name, coeffs, d_coeffs in calibration_coefficients
]
ufn.save_calibration_coefficients(calibration_coefficients, "calibration_coefficients.csv")
