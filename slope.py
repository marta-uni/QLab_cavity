import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import find_peaks
import piecewise_regression

#####################################################################################################
# section with file specific values

title_list = [
    f'calibrated_error{str(i).zfill(5)}_filtered' for i in range(1, 7)]
title_list = title_list[:4] + title_list[5:]
'''# excluding files with "throw out note" (8 to 10) and error5 file that has a bad error signal
title_list = title_list[:4] + title_list[5:7] + title_list[10:]'''

'''ind_range = [200, 200, 150, 100, 250, 200, 200,
             200, 200, 200, 200, 150, 200, 200]'''
ind_range = [200, 200, 200, 200, 200, 200, 200]
min_height = 0.04
min_distance = 1000

figure_path = 'data_non_confocal/figures/slope'
file_path = 'data_non_confocal/clean_data'
os.makedirs(figure_path, exist_ok=True)
os.makedirs(file_path, exist_ok=True)

#####################################################################################################

D = []
d_D = []

'''mod_f = [50, 40, 30, 20, 10, 5, 40, 40, 40, 40, 40, 40, 40, 40]
mod_ampl = [510, 310, 250, 150, 30, 8, 360, 300, 250, 200, 140, 110, 400, 450]'''

mod_f = [50, 40, 30, 20, 10]
mod_ampl = [510, 310, 250, 150, 30]

for i, title in enumerate(title_list):
    print(title)
    data = pd.read_csv(f'data_non_confocal/clean_data/{title}.csv')

    freq = data['frequencies'].to_numpy()
    freq_spacing = np.mean(np.diff(freq))
    reflection = data['reflection'].to_numpy()
    transmission = data['transmission'].to_numpy()

    if title == 'calibrated_error00018_filtered':
        min_height = 0.025
        min_distance = 200

    peaks_indices, _ = find_peaks(
        data['transmission'], height=min_height, distance=min_distance)
    if len(peaks_indices) > 1:
        peaks_indices = np.delete(peaks_indices, [1])
    elif len(peaks_indices) == 0:
        continue

    f_peaks = freq[peaks_indices]
    refl_peaks = reflection[peaks_indices]
    trans_peaks = transmission[peaks_indices]

    '''plotting, if necessary'''

    fname = os.path.join(figure_path, f'calib_refl_plot_{title}.png')

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].scatter(freq, reflection, label='Data', color='blue', s=8)
    axs[0].scatter(f_peaks, refl_peaks, label='Peaks',
                   color='red', marker='x')
    axs[0].set_xlabel('Frequency [MHz]')
    axs[0].set_ylabel('Reflection Signal [V]')
    axs[0].axvline(f_peaks[0]-freq_spacing*ind_range[i], color='black')
    axs[0].axvline(f_peaks[0]+freq_spacing*ind_range[i], color='black')
    axs[0].legend()
    axs[0].grid()

    axs[1].scatter(freq, transmission, label='Data', color='blue', s=8)
    axs[1].scatter(f_peaks, trans_peaks, label='Peaks',
                   color='red', marker='x')
    axs[1].set_xlabel('Frequency [MHz]')
    axs[1].set_ylabel('Transmission Signal [V]')
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

    peak_region = data.iloc[peaks_indices[0] -
                            ind_range[i]:peaks_indices[0]+ind_range[i] + 1]
    refl_restricted = peak_region['reflection'].to_numpy()
    f_restricted = peak_region['frequencies'].to_numpy()

    pw_fit = piecewise_regression.Fit(
        f_restricted, refl_restricted, n_breakpoints=2)
    pw_results = pw_fit.get_results()
    print(
        f'{pw_results["estimates"]["alpha2"]["estimate"]} +/- {pw_results["estimates"]["alpha2"]["se"]}')

    D.append(pw_results["estimates"]["alpha2"]["estimate"])
    d_D.append(pw_results["estimates"]["alpha2"]["se"])

    fname = os.path.join(figure_path, f'calib_pw_{title}.png')

    plt.figure()
    pw_fit.plot_data(color="blue", s=20, label='Data')
    pw_fit.plot_fit(color="orange", linewidth=2, label='Fit')
    pw_fit.plot_breakpoints()
    pw_fit.plot_breakpoint_confidence_intervals()
    plt.scatter(f_peaks, refl_peaks, label='Peaks', color='red', marker='x')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Reflection Signal [V]')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

df = pd.DataFrame({
    'title': title_list,
    'D': D,
    'd_D': d_D,
    'mod_f': mod_f,
    'mod_ampl': mod_ampl
})

filename = os.path.join(file_path, 'calib_slopes.csv')
df.to_csv(filename, index=False)
