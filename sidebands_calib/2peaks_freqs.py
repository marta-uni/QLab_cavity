import pandas as pd
import numpy as np
import os
import functions as fn
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

#####################################################################################################
# generate filenames

title_list = [f'error{str(i).zfill(5)}' for i in range(1, 19)]
#titles_list = title_list[:4] + title_list[10:15] + title_list[16:18]
titles_list = title_list[:4] + title_list[5:7] + title_list[10:15] + title_list[16:18]
print(titles_list)
os.makedirs(f"calibration/figures/residuals", exist_ok=True)

########################################################################################################

def sum_of_lorentzians_off(x, *params):
    """
    Sum of multiple Lorentzian functions with a single global offset.
    x: input array
    params: [A1, x0_1, gamma_1, ..., A_n, x0_n, gamma_n, offset]
    Returns the sum of the Lorentzian peaks plus a constant offset.
    """
    n_peaks = (len(params) - 1) // 3  # Last parameter is the global offset
    result = np.zeros_like(x)
    for i in range(n_peaks):
        A = params[3 * i]     # Amplitude
        x0 = params[3 * i + 1]  # Center
        gamma = params[3 * i + 2]  # Width
        result += A * (gamma**2 / ((x - x0)**2 + gamma**2))
    
    offset = params[-1]  # Single global offset
    return result + offset


#####################################################################################################

min_heights = [0.01, 0.01, 0.005, 0.005, 0.01, 0.003, 0.01, 0.008, 0.006, 0.004, 0.01, 0.01, 0.01 ]
omegas = [50, 40, 30, 20, 40, 40, 40, 40, 40, 40, 40] #MHz

#####################################################################################################
file_path = "fitted_parameters.csv"  # Replace with your file's path
data = pd.read_csv(file_path)

# Filter the data for files with exactly 3 peaks
files_with_three_peaks = data.groupby("file_name").filter(lambda x: len(x) == 3)
print(len(files_with_three_peaks))

# Extract parameters into arrays for files with 3 peaks
file_names = files_with_three_peaks["file_name"].unique()
peak_A = []

#these arrays now include "subarrays" of triplets of peak parameters, stored as ufloats
for file in file_names:
    file_data = files_with_three_peaks[files_with_three_peaks["file_name"] == file]
    peak_A.append(file_data["A"].values)
    #peak_offset.append([ufloat(o, do) for o, do in zip(file_data["offset"].values, file_data["offset_uncertainty"].values)])

# Convert lists to numpy arrays
peak_A = np.array(peak_A, dtype=object)

# Create the output file
output_file = 'fitted_parameters_freqs.csv'

with open(output_file, 'w') as file:
    # Write the header
    file.write('file_name,peak,A,x0,gamma,A_uncertainty,x0_uncertainty,gamma_uncertainty\n')
    # Loop through your dataset
    for i, (title, h) in enumerate(zip(titles_list, min_heights)):
        # Load the filtered data
        data = pd.read_csv(f'clean_data/calibrated_{title}_filtered.csv')
        print(f'opened {title}')
        frequencies = data['frequencies'].to_numpy()
        transmitted = data['transmission'].to_numpy()

        # Detect peaks
        # Determine peak threshold based on index in titles_list
        #peak_threshold = 120 if title == titles_list[4] else 265
        peak_threshold = 265

        # Detect peaks with the modified threshold
        title_index = np.where(file_names == title)[0][0]  # Find the index of `title` in `file_names`
        ypeaks = peak_A[title_index] 
        xpeaks = [0, omegas[i], omegas[i] *2]
        peak_widths = [4,4,4]
        # Prepare initial guesses for the multi-peak fitting

        # Special handling for error0006 to merge middle peaks
        '''if title == titles_list[4] and len(xpeaks) == 4 :
            # Assume first and last peaks remain unchanged
            print('error6 had 4 peaks')
            xpeaks = [xpeaks[0], (xpeaks[1] + xpeaks[2]) / 2, xpeaks[3]+5]
            ypeaks = [ypeaks[0], (ypeaks[1] + ypeaks[2]) / 2, ypeaks[3]]
            peak_widths = [4,4,4]'''

        # Construct initial guess list
        initial_guess = []
        for xpeak, ypeak, width in zip(xpeaks, ypeaks, peak_widths):
            initial_guess.extend([ypeak, xpeak, width])

        # Append a single global offset guess (ensuring it is positive)
        initial_guess.append(max(min(transmitted), 0.001))
        print(f'{title} : ', initial_guess)
        # Define bounds: Amplitudes, gamma, and offset should be > 0
        lower_bounds = [-5] * (len(initial_guess) - 1) + [0]  # Last value is for offset
        upper_bounds = [np.inf] * len(initial_guess)  # No strict upper bounds

        # Modify lower bound for middle peak position (index 4 in the parameter list)
        '''if title == titles_list[4]:
            lower_bounds[7] = xpeaks[2]-10 # Ensure one of the peak positions is above 0.795'''        
        
        # Perform curve fitting with bounds
        popt, pcov = curve_fit(
            sum_of_lorentzians_off,
            frequencies,
            transmitted,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=50000
        )

        # Extract and print the offset
        offset = popt[-1]  # Last parameter is the offset
        offset_uncertainty = np.sqrt(pcov[-1, -1]) if pcov.shape == (len(popt), len(popt)) else np.nan

        print(f"Global offset for {title}: {offset:.5f} ± {offset_uncertainty:.5f}")
        print(f"n peaks for {title}:{(len(initial_guess) - 1)/3}")

        try:
            # Perform the multi-peak fit using curve_fit
            popt, pcov = curve_fit(sum_of_lorentzians_off, frequencies, transmitted, p0=initial_guess, maxfev=10000)

            # Calculate the uncertainties (square root of diagonal elements of the covariance matrix)
            uncertainties = np.sqrt(np.diag(pcov))

            ########################WRITE ON FILE####################################

            # Loop through the peaks and write the results
            for i, (xpeak, ypeak) in enumerate(zip(xpeaks, ypeaks)):
                # Extract the parameters and uncertainties for the current peak
                A = popt[3 * i]     # Amplitude
                x0 = popt[3 * i + 1]  # Center
                gamma = popt[3 * i + 2]  # Width

                A_uncertainty = uncertainties[3 * i]     # Uncertainty in amplitude
                x0_uncertainty = uncertainties[3 * i + 1]  # Uncertainty in center
                gamma_uncertainty = uncertainties[3 * i + 2]  # Uncertainty in width

                # Write the parameters and uncertainties to the file
                file.write(f'{title},{i+1},{A},{x0},{gamma},{A_uncertainty},{x0_uncertainty},{gamma_uncertainty}\n')
            

            ########################PLOT####################################

            # Generate the fitted curve
            x_fit = np.linspace(frequencies.min(), frequencies.max(), 1000)
            y_fit = sum_of_lorentzians_off(x_fit, *popt)
                        # Initialize the plot for the multi-peak fit
            plt.figure(figsize=(10, 6))
            plt.scatter(frequencies, transmitted, label='Original Data', color='blue', alpha=0.6, marker='.')
            # Plot the fitted multi-peak curve
            plt.plot(x_fit, y_fit, label='Multi-peak Fit', color='red', linestyle='--')


            # Finalize the plot
            plt.xlabel('Freq (MHz)')
            plt.ylabel('Transmission')
            plt.title(f'Multi-Peak Lorentzian Fits for {title}')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.yscale('log')
            plt.ylim(0.0001, None)  # Lower limit at 10^-3; no upper limit specified

            # Save the multi-peak fit plot
            output_dir = 'calibration/figures/multi_peak_fits'
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f'{output_dir}/{title}_multipeak_fits.png')
            plt.close()

            # --- RESIDUALS PLOTTING ---
            # Calculate the residuals
            residuals = transmitted - sum_of_lorentzians_off(frequencies, *popt)

            # Plot the residuals
            plt.figure(figsize=(10, 6))
            plt.scatter(frequencies, residuals, color='blue', alpha=0.6, marker='.')
            plt.axhline(0, color='red', linestyle='--')  # Zero line
            plt.xlabel('Freq (MHz)')
            plt.ylabel('Residuals (Transmission - Fit)')
            plt.title(f'Residuals for Multi-Peak Lorentzian Fit - {title}')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()

            # Save the residuals plot
            residuals_dir = 'calibration/figures/residuals'
            os.makedirs(residuals_dir, exist_ok=True)
            plt.savefig(f'{residuals_dir}/{title}_residuals.png')
            plt.close()

        except RuntimeError as e:
            print(f"Failed to fit multi-peak model for {title}: {e}")
