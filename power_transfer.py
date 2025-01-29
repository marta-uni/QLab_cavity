import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit

data_folder = 'data_non_confocal/clean_data'

peaks_data = pd.read_csv(f'{data_folder}/fitted_parameters.csv')
peaks_data = peaks_data.iloc[14:]

grouped_peaks = {label: group for label,
                 group in peaks_data.groupby(peaks_data['file_name'])}

A_0 = []
dA_0 = []
A_s = []
dA_s = []
A_c = []
dA_c = []

for label, df in grouped_peaks.items():
    total_sum = sum(unp.uarray(
        df['A'].to_numpy(), df['A_uncertainty'].to_numpy()))
    A_0.append(total_sum.n)
    dA_0.append(total_sum.std_dev)
    
    if label == 'error00016':
        A_s.append(0)
        dA_s.append(0)
        A_c.append(total_sum.n)
        dA_c.append(total_sum.std_dev)
        continue

    carrier = df[df['peak'] == 2]['A'].values[0]
    d_carrier = df[df['peak'] == 2]['A_uncertainty'].values[0]

    subset = df[df['peak'].isin([1, 3])]
    sb = subset['A'].values
    sb_unc = subset['A_uncertainty'].values

    u_sb = [ufloat(val, err) for val, err in zip(sb, sb_unc)]
    sb_mean = sum(u_sb)/2

    A_s.append(sb_mean.n)
    dA_s.append(sb_mean.std_dev)
    A_c.append(carrier)
    dA_c.append(d_carrier)

x_unc_c0 = unp.uarray(A_c, dA_c) / unp.uarray(A_0, dA_0)
x_unc_sc = unp.uarray(A_s, dA_s) / unp.uarray(A_c, dA_c)

slope_data = pd.read_csv(f'{data_folder}/calib_slopes_power.csv')

# D units are V/MHz

D_unc = unp.uarray(
    np.abs(slope_data['D'].to_numpy()), slope_data['d_D'].to_numpy())

y_unc = D_unc / unp.uarray(A_0, dA_0)

y_data = unp.nominal_values(y_unc)
dy_data = unp.std_devs(y_unc)
x_c0 = unp.nominal_values(x_unc_c0)
dx_c0 = unp.std_devs(x_unc_c0)
x_sc = unp.nominal_values(x_unc_sc)
dx_sc = unp.std_devs(x_unc_sc)


def circle(x, a):
    return a * np.sqrt(x * (1 - x))


def other_func(x, a):
    factor = 1/(1+2*x)
    return a * np.sqrt(factor * (1 - factor))


xplot = np.linspace(0, 1, 500)
yplot_c0 = circle(xplot, 3)
yplot_sc = other_func(xplot, 3)

plt.figure()
plt.errorbar(x_c0, y_data, yerr=dy_data, xerr=dx_c0,
             ls='', label='Data', color='blue', fmt='.')
plt.plot(xplot, yplot_c0, label='Expected behaviour', color='red', linewidth=2)
plt.xlabel('Ac/A0')
plt.ylabel('D/A0 [MHz^-1]')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('data_non_confocal/figures/power_transfer_c0.png')

plt.figure()
plt.errorbar(x_sc, y_data, yerr=dy_data, xerr=dx_c0,
             ls='', label='Data', color='blue', fmt='.')
plt.plot(xplot, yplot_sc, label='Expected behaviour', color='red', linewidth=2)
plt.xlabel('As/Ac')
plt.ylabel('D/A0 [MHz^-1]')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('data_non_confocal/figures/power_transfer_sc.png')
plt.show()
