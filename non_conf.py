import pandas as pd
import functions as fn
import fit_peaks as fp
import numpy as np
import os

c = 3e8  # speed of light
R = 50e-3  # curvature radius

#####################################################################################################
# section with file specific values

title = 'calib00001'
path = f'data_non_confocal/clean_data/{title}.csv'
os.makedirs(f"data_non_confocal/figures/fsr", exist_ok=True)

#####################################################################################################

'''read data'''
data = pd.read_csv(path)

timestamps = data['timestamp'].to_numpy()
transmission = data['transmission'].to_numpy()
volt_piezo = data['volt_piezo'].to_numpy()

'''find free spectral range (in voltage units) from peaks'''

# find peaks and widths with just the find_peak function (only for plotting)
xpeaks, ypeaks, peak_widths, indices = fn.peaks(
    volt_piezo, transmission, 0.01, 100, True)

# find peaks and widths fitting lorentzians
lor, cov = fp.fit_peaks(volt_piezo, transmission, 0.01, 100)

x0_list = []
A_list = []
gamma_list = []
dx0 = []
dA = []
dgamma = []

for popt, pcov in zip(lor, cov):
     A_list.append(popt[0])
     x0_list.append(popt[1])
     gamma_list.append(popt[2])
     dA.append(np.sqrt(pcov[0, 0]))
     dx0.append(np.sqrt(pcov[1, 1]))
     dgamma.append(np.sqrt(pcov[2, 2]))

x0_list = np.array(x0_list)
A_list = np.array(A_list)
gamma_list = np.array(gamma_list)
dx0 = np.array(dx0)
dA = np.array(dA)
dgamma = np.array(dgamma)

fn.plot_piezo_laser_fit(volt_piezo, transmission, file_name=f'data_non_confocal/figures/fsr/piezo_peaks_{title}.png', A=A_list,
                         x0=x0_list, gamma=gamma_list, xpeaks=xpeaks, ypeaks=ypeaks, width=peak_widths, save=True)

# zoomed picture
vp_z = volt_piezo[indices[4]-400:indices[4]+400]
tr_z = transmission[indices[4]-400:indices[4]+400]
a_z = A_list[3:6]
x0_z = x0_list[3:6]
g_z = gamma_list[3:6]
x_z = xpeaks[3:6]
y_z = ypeaks[3:6]
w_z = peak_widths[3:6]

fn.plot_piezo_laser_fit(vp_z, tr_z, file_name=f'data_non_confocal/figures/fsr/piezo_peaks_closeup_{title}.png', A=a_z,
                         x0=x0_z, gamma=g_z, xpeaks=x_z, ypeaks=y_z, width=w_z, save=True)

ind_ = np.argsort(ypeaks)[-3:]
tem00_indices = sorted(ind_)
tem00_x0 = x0_list[tem00_indices]
tem00_dx0 = dx0[tem00_indices]

print(f'FSR in volt: {(tem00_x0[-1]-tem00_x0[0])/2} +/- {0.5*np.sqrt(tem00_dx0[-1]**2+tem00_dx0[0]**2)} V')
