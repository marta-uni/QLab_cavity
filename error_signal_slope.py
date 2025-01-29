import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit

data_folder = 'data_non_confocal/clean_data'

peaks_data = pd.read_csv(f'{data_folder}/fitted_parameters_freqs.csv')

gamma_values = peaks_data.loc[peaks_data['peak'] == 2, 'gamma']
gamma_unc = peaks_data.loc[peaks_data['peak'] == 2, 'gamma_uncertainty']

hwhm = sum(unp.uarray(gamma_values.to_numpy(),
           gamma_unc.to_numpy())) / len(gamma_values.index)

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

# factor_list = [unp.sqrt(c * s) for c, s in zip(unp.uarray(A_s, dA_s), unp.uarray(A_c, dA_c))]

factor_arr = unp.sqrt(unp.uarray(A_s, dA_s) * unp.uarray(A_c, dA_c))

slope_data = pd.read_csv(f'{data_folder}/calib_slopes.csv')

mod_f = slope_data['mod_f'].to_numpy()
D_unc = unp.uarray(np.abs(slope_data['D'].to_numpy()), slope_data['d_D'].to_numpy())

y_unc = D_unc / factor_arr

y_fit = unp.nominal_values(y_unc)
dy_fit = unp.std_devs(y_unc)


def linear(x, a, b):
    return a * x + b


popt, pcov = curve_fit(linear, xdata=mod_f, ydata=y_fit, p0=[
                       0, np.mean(y_fit)], sigma=dy_fit, absolute_sigma=True)

print(f'Fit result: a = {popt[0]} +/- {np.sqrt(pcov[0, 0])} MHz^-1')
print(f'b = {popt[1]} +/- {np.sqrt(pcov[1, 1])}')

xplot = np.linspace(mod_f[0], mod_f[-1], 100)

plt.figure()
plt.errorbar(mod_f, y_fit, dy_fit, ls='', label='Data', color='blue', fmt='.')
plt.plot(xplot, linear(xplot, *popt),
         label='Linear Fit', color='red', linewidth=2)
plt.axvline(hwhm.n, color='black', label='HWHM')
y_min, y_max = plt.gca().get_ylim()
plt.fill_betweenx(y=np.linspace(y_min, y_max, 100), x1=hwhm.n -
                  hwhm.std_dev, x2=hwhm.n + hwhm.std_dev, color='grey', alpha=0.3)
plt.xlabel('Frequency modulation [MHz]')
plt.ylabel('D/sqrt(PcPs)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('data_non_confocal/figures/error_signal_slope.png')
plt.show()
