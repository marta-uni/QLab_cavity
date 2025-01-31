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

data6 = pd.read_csv(f'{data_folder}/fitted_parameters_freq6.csv')
peaks_data = pd.concat([peaks_data, data6], ignore_index=True)


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
mod_ampl = slope_data['mod_ampl'].to_numpy()/1000

D_values = np.abs(slope_data['D'].to_numpy())
d_D = slope_data['d_D'].to_numpy()

plt.figure()
plt.errorbar(mod_f, D_values, d_D, ls='', label='Data', color='blue', fmt='.', capsize=5)
plt.axvline(hwhm.n, color='black', label=f'FWHM: {hwhm} MHz')
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

beta = np.array([fsolve(partial(extract_beta, asac=ratio.nominal_value), 0.9)[0]
        for ratio in As_Ac])

beta_plus = np.array([fsolve(partial(extract_beta, asac=(ratio.nominal_value + ratio.std_dev)), 1)[0]
             for ratio in As_Ac])

beta_minus = np.array([fsolve(partial(extract_beta, asac=(ratio.nominal_value - ratio.std_dev)), 1)[0]
              for ratio in As_Ac])

beta_plus = np.abs(beta_plus - beta)
beta_minus = np.abs(beta_minus - beta)

d_beta = [max(a, b) for a, b in zip(beta_minus, beta_plus)]
beta = unp.uarray(beta, d_beta)

v_pi = mod_ampl * np.pi / beta


def linear(x, a, b):
    return a * x + b


popt, pcov = curve_fit(linear, mod_f, unp.nominal_values(
    v_pi), sigma=unp.std_devs(v_pi), absolute_sigma=True)
x_line = np.linspace(min(mod_f), max(mod_f))

plt.figure()
plt.errorbar(mod_f, unp.nominal_values(v_pi), yerr=unp.std_devs(
    v_pi), color='blue', marker='.', ls='', label='data', capsize=5)
plt.plot(x_line, linear(x_line, *popt), color='red',
         linewidth=2, label='linear fit')
plt.xlabel('Frequency modulation [MHz]')
plt.ylabel(r'$\text{V}_{\pi}$ [V]')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('data_non_confocal/figures/v_pi_estimate.png')
plt.show()
