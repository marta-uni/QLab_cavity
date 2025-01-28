import pandas as pd
import functions as fn
import fit_peaks as fp
import numpy as np
import os

#####################################################################################################
# section with file specific values

title_list = [f'bessel{str(i).zfill(5)}' for i in range(0, 22)]
# removing extra files
title_list = title_list[:8] + title_list[9:20] + title_list[21:]

min_height = [0.032, 0.032, 0.0055, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
              0.002, 0.002, 0.002, 0.002, 0.003, 0.003, 0.003, 0.003, 0.003]

lb = [1.65, 1.55, 1.58, 1.55, 1.55, 1.55, 1.55, 1.5, 1.5, 1.45, 1.45, 1.42, 1.3,
      1.41, 1.45, 1.43, 1.445, 1.375, 1.45, 1.46]

ub = [1.75, 1.65, 1.67, 1.7, 1.7, 1.62, 1.65, 1.65, 1.62, 1.6, 1.6, 1.6, 1.45,
      1.53, 1.56, 1.55, 1.56, 1.52, 1.57, 1.575]

os.makedirs(f"data_bessel/figures/find_sidebands", exist_ok=True)
os.makedirs(f"data_bessel/figures/find_sidebands/residuals", exist_ok=True)

ind_range = 200

figure_name = 'find_sb_log'
log = True

#####################################################################################################

A_peaks = np.array([])
d_A_peaks = np.array([])
x0_peaks = np.array([])
d_x0_peaks = np.array([])
gamma_peaks = np.array([])
d_gamma_peaks = np.array([])
off_peaks = np.array([])
d_off_peaks = np.array([])
title_peaks = np.array([])
type_peaks = np.array([])


for i, title in enumerate(title_list):
    data = pd.read_csv(f'data_bessel/clean_data/{title}.csv')
    print(title)
    volt_piezo = data['volt_piezo'].to_numpy()
    volt_laser = data['transmission'].to_numpy()
    
    mask = (volt_piezo > lb[i]) & (volt_piezo < ub[i])
    volt_piezo = volt_piezo[mask]
    volt_laser = volt_laser[mask]

    xpeaks, ypeaks, peak_widths = fn.peaks(
        volt_piezo, volt_laser, min_height[i], 200)
    lor, cov = fp.fit_peaks_leonardi(
        volt_piezo, volt_laser, xpeaks, peak_widths, ind_range)

    x0_list = []
    A_list = []
    gamma_list = []
    off_list = []
    dA = []
    dx0 = []
    dgamma = []
    doff = []

    for popt, pcov in zip(lor, cov):
        A_list.append(popt[0])
        x0_list.append(popt[1])
        gamma_list.append(popt[2])
        off_list.append(popt[3])
        dA.append(np.sqrt(pcov[0, 0]))
        dx0.append(np.sqrt(pcov[1, 1]))
        dgamma.append(np.sqrt(pcov[2, 2]))
        doff.append(np.sqrt(pcov[3, 3]))

    x0_list = np.array(x0_list)
    A_list = np.array(A_list)
    gamma_list = np.array(gamma_list)
    off_list = np.array(off_list)
    dA = np.array(dA)
    dx0 = np.array(dx0)
    dgamma = np.array(dgamma)
    doff = np.array(doff)

    title_peaks = np.append(title_peaks, [title]*len(xpeaks))
    A_peaks = np.append(A_peaks, A_list)
    x0_peaks = np.append(x0_peaks, x0_list)
    gamma_peaks = np.append(gamma_peaks, gamma_list)
    off_peaks = np.append(off_peaks, off_list)
    d_A_peaks = np.append(d_A_peaks, dA)
    d_x0_peaks = np.append(d_x0_peaks, dx0)
    d_gamma_peaks = np.append(d_gamma_peaks, dgamma)
    d_off_peaks = np.append(d_off_peaks, doff)
    

    if len(xpeaks) == 1:
        type_peaks = np.append(type_peaks, ['carrier'])
    elif len(xpeaks) == 2:
        type_peaks = np.append(type_peaks, ['carrier', 'sb_1'])
    elif len(xpeaks) == 3:
        type_peaks = np.append(type_peaks, ['sb_1', 'carrier', 'sb_1'])
    elif len(xpeaks) == 5:
        type_peaks = np.append(type_peaks, ['sb_2', 'sb_1', 'carrier', 'sb_1', 'sb_2'])

    fp.plot_piezo_laser_fit_leonardi(volt_piezo, volt_laser, file_name=f'data_bessel/figures/find_sidebands/{figure_name}_{title}.png', A=A_list,
                                     x0=x0_list, gamma=gamma_list, off=off_list, xpeaks=xpeaks, ypeaks=ypeaks, ind_range=ind_range, save=True, log=log)

output_file = f'data_bessel/clean_data/bessel_sidebands.csv'
df = pd.DataFrame()
df['title'] = title_peaks
df['type'] = type_peaks
df['lor_A'] = A_peaks
df['lor_mean'] = x0_peaks
df['lor_gamma'] = gamma_peaks
df['lor_off'] = off_peaks
df['lor_d_A'] = d_A_peaks
df['lor_d_mean'] = d_x0_peaks
df['lor_d_gamma'] = d_gamma_peaks
df['lor_d_off'] = d_off_peaks

df.to_csv(output_file, index=False)
print(f"Data saved to {output_file}")
