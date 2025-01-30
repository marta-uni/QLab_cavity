import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy.special import jv
from scipy.optimize import fsolve, curve_fit
from functools import partial

data_folder = 'data_non_confocal/clean_data'

peaks_data = pd.read_csv(f'{data_folder}/fitted_parameters_freqs.csv')
peaks_data = peaks_data.iloc[:12]

print(peaks_data)

gamma_values = peaks_data.loc[peaks_data['peak'] == 2, 'gamma']
gamma_unc = peaks_data.loc[peaks_data['peak'] == 2, 'gamma_uncertainty']

hwhm = sum(unp.uarray(gamma_values.to_numpy(),
           gamma_unc.to_numpy())) / len(gamma_values.index) * 2

grouped_peaks = {label: group for label,
                 group in peaks_data.groupby(peaks_data['file_name'])}

A_s = []
dA_s = []
A_c = []
dA_c = []

for label, df in grouped_peaks.items():
    carrier = df[df['peak'] == 2]['A'].values[0]
    d_carrier = df[df['peak'] == 2]['A_uncertainty'].values[0]

    subset = df[df['peak'].isin([1, 3])]
    sb = subset['A'].values
    sb_unc = subset['A_uncertainty'].values

    u_sb = [ufloat(val, err) for val, err in zip(sb, sb_unc)]
    sb_mean = sum(u_sb)/2

    A_s.append(sb_mean.nominal_value)
    dA_s.append(sb_mean.std_dev)
    A_c.append(carrier)
    dA_c.append(d_carrier)

slope_data = pd.read_csv(f'{data_folder}/calib_slopes.csv')

mod_f = slope_data['mod_f'].to_numpy()

D_values = np.abs(slope_data['D'].to_numpy())
d_D = slope_data['d_D'].to_numpy()


def linear(x, a, b):
    return a * x + b


popt, pcov = curve_fit(linear, xdata=mod_f, ydata=D_values, p0=[
                       0, np.mean(D_values)], sigma=d_D, absolute_sigma=True)

print(f'Fit result: a = {popt[0]} +/- {np.sqrt(pcov[0, 0])} V*MHz^-2')
print(f'b = {popt[1]} +/- {np.sqrt(pcov[1, 1])} V')

xplot = np.linspace(mod_f[0], mod_f[-1], 100)

plt.figure()
plt.errorbar(mod_f, D_values, d_D, ls='', label='Data', color='blue', fmt='.')
plt.plot(xplot, linear(xplot, *popt),
         label='Linear Fit', color='red', linewidth=2)
plt.axvline(hwhm.n, color='black', label=f'2*HWHM: {hwhm} MHz')
y_min, y_max = plt.gca().get_ylim()
plt.fill_betweenx(y=np.linspace(y_min, y_max, 100), x1=hwhm.n -
                  hwhm.std_dev, x2=hwhm.n + hwhm.std_dev, color='grey', alpha=0.3)
plt.xlabel('Frequency modulation [MHz]')
plt.ylabel('D [V/MHz]')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('data_non_confocal/figures/error_signal_slope.png')

As_Ac = unp.uarray(A_s, dA_s) / unp.uarray(A_c, dA_c)


def extract_beta(x, asac):
    return (jv(1, x))**2 / (jv(0, x))**2 - asac


x = np.linspace(0, 2, 100)
y = (jv(1, x))**2 / (jv(0, x))**2

beta = [fsolve(partial(extract_beta, asac=ratio.nominal_value), 0.9)[0]
        for ratio in As_Ac]

plt.figure()
plt.plot(mod_f, beta, color='red', linewidth=2, marker='o')
plt.xlabel('Frequency modulation [MHz]')
plt.ylabel(r'$\beta$')
plt.grid()
plt.tight_layout()
plt.savefig('data_non_confocal/figures/beta_bad_estimate.png')
plt.show()
