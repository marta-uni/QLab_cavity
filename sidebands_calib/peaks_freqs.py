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
titles_list = title_list[:4] + title_list[10:15] + title_list[16:18]
print(titles_list)
os.makedirs(f"peaks_frequencies/figures/peaks", exist_ok=True)
os.makedirs(f"peaks_frequencies/figures/residuals", exist_ok=True)

########################################################################################################

# Define functions
def lorentzian(x, A, x0, gamma, offset):
    """
    Lorentzian function with amplitude A, center x0, width gamma, and offset.
    """
    return A * (gamma**2 / ((x - x0)**2 + gamma**2)) + offset

def sum_of_lorentzians(x, *params):
    """
    Sum of multiple Lorentzian functions.
    x: input array
    params: [A1, x0_1, gamma_1, offset_1, A2, x0_2, gamma_2, offset_2, ..., A_n, x0_n, gamma_n, offset_n]
    Returns the sum of the Lorentzian peaks.
    """
    n_peaks = len(params) // 4  # Each peak has 4 parameters: A, x0, gamma, offset
    result = np.zeros_like(x)
    for i in range(n_peaks):
        A = params[4 * i]     # Amplitude
        x0 = params[4 * i + 1]  # Center
        gamma = params[4 * i + 2]  # Width
        offset = params[4 * i + 3]  # Offset
        result += A * (gamma**2 / ((x - x0)**2 + gamma**2)) + offset
    return result

#####################################################################################################

min_heights = [0.01, 0.01, 0.005, 0.005, 0.07, 0.07, 0.003, 0.01, 0.008, 0.006, 0.004, 0.01, 0.01, 0.01 ]

#####################################################################################################
# Create the output file
output_file = 'fitted_parameters_freqs.csv'


with open(output_file, 'w') as file:
    # Write the header
    file.write('file_name,peak,A,x0,gamma,offset,A_uncertainty,x0_uncertainty,gamma_uncertainty,offset_uncertainty\n')
    # Loop through your dataset
    for title, h in zip(titles_list, min_heights):
        # Load the filtered data
        data = pd.read_csv(f'clean_data/calibrated_{title}_filtered.csv')
        frequencies = data['frequencies'].to_numpy()
        transmitted = data['transmission'].to_numpy()

        # Detect peaks
        xpeaks, ypeaks, peak_widths = fn.peaks(frequencies, transmitted, h, 270)

        # Initialize the plot for the multi-peak fit
        plt.figure(figsize=(10, 6))
        plt.scatter(frequencies, transmitted, label='Original Data', color='blue', alpha=0.6, marker='.')

        # Prepare initial guesses for the multi-peak fitting
        initial_guess = []
        for xpeak, ypeak, width in zip(xpeaks, ypeaks, peak_widths):
            initial_guess.extend([ypeak, xpeak, width, min(transmitted)])

        try:
            # Perform the multi-peak fit using curve_fit
            popt, pcov = curve_fit(sum_of_lorentzians, frequencies, transmitted, p0=initial_guess, maxfev=10000)

            # Calculate the uncertainties (square root of diagonal elements of the covariance matrix)
            uncertainties = np.sqrt(np.diag(pcov))

            # Generate the fitted curve
            x_fit = np.linspace(frequencies.min(), frequencies.max(), 1000)
            y_fit = sum_of_lorentzians(x_fit, *popt)

            # Plot the fitted multi-peak curve
            plt.plot(x_fit, y_fit, label='Multi-peak Fit', color='red', linestyle='--')

            # Loop through the peaks and write the results
            for i, (xpeak, ypeak) in enumerate(zip(xpeaks, ypeaks)):
                # Extract the parameters and uncertainties for the current peak
                A = popt[4 * i]     # Amplitude
                x0 = popt[4 * i + 1]  # Center
                gamma = popt[4 * i + 2]  # Width
                offset = popt[4 * i + 3]  # Offset

                A_uncertainty = uncertainties[4 * i]     # Uncertainty in amplitude
                x0_uncertainty = uncertainties[4 * i + 1]  # Uncertainty in center
                gamma_uncertainty = uncertainties[4 * i + 2]  # Uncertainty in width
                offset_uncertainty = uncertainties[4 * i + 3]  # Uncertainty in offset

                # Write the parameters and uncertainties to the file
                file.write(f'{title},{i+1},{A},{x0},{gamma},{offset},{A_uncertainty},{x0_uncertainty},{gamma_uncertainty},{offset_uncertainty}\n')

            # Finalize the plot
            plt.xlabel('Frequencies (MHz)')
            plt.ylabel('Transmission (V)')
            plt.title(f'Multi-Peak Lorentzian Fits for {title}')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.yscale('log')
            plt.ylim(0.0013, None)  # Lower limit at 10^-3; no upper limit specified

            # Save the multi-peak fit plot
            output_dir = 'peaks_frequencies/figures/multi_peak_fits'
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f'{output_dir}/{title}_multipeak_freqfits.png')
            plt.close()

            # --- RESIDUALS PLOTTING ---
            # Calculate the residuals
            residuals = transmitted - sum_of_lorentzians(frequencies, *popt)

            # Plot the residuals
            plt.figure(figsize=(10, 6))
            plt.scatter(frequencies, residuals, color='blue', alpha=0.6, marker='.')
            plt.axhline(0, color='red', linestyle='--')  # Zero line
            plt.xlabel('Frequencies (MHz)')
            plt.ylabel('Residuals (Transmission - Fit) (V)')
            plt.title(f'Residuals for Multi-Peak Lorentzian Fit - {title}')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()

            # Save the residuals plot
            residuals_dir = 'peaks_frequencies/figures/residuals'
            os.makedirs(residuals_dir, exist_ok=True)
            plt.savefig(f'{residuals_dir}/{title}_residuals_freq.png')
            plt.close()

        except RuntimeError as e:
            print(f"Failed to fit multi-peak model for {title}: {e}")
