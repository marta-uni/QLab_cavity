import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit
from scipy import odr

data_folder = 'data_non_confocal/clean_data'

peaks_data = pd.read_csv(f'{data_folder}/fitted_parameters.csv')
peaks_data = peaks_data.iloc[16:]
peaks_data = peaks_data.drop(31)

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

D_values = np.abs(slope_data['D'].to_numpy())
D_unc = slope_data['d_D'].to_numpy()

mask = (x_unc_c0 > 0.7)
x_unc_c0 = x_unc_c0[mask]
x_unc_sc = x_unc_sc[mask]
D_values = D_values[mask]
D_unc = D_unc[mask]

x_c0 = unp.nominal_values(x_unc_c0)
dx_c0 = unp.std_devs(x_unc_c0)
x_sc = unp.nominal_values(x_unc_sc)
dx_sc = unp.std_devs(x_unc_sc)


def circle(par, x):
    return par[0] * np.sqrt(x * (1 - x))


def other_func(par, x):
    ratio = 1/(1+2*x)
    return par[0] * np.sqrt(ratio * (1 - ratio))


circle_model = odr.Model(circle)
other_model = odr.Model(other_func)
aca0_data = odr.RealData(x_c0, D_values, dx_c0, D_unc)
asac_data = odr.RealData(x_sc, D_values, dx_sc, D_unc)
odr_circle = odr.ODR(aca0_data, circle_model, beta0=[0.4])
odr_other = odr.ODR(asac_data, other_model, beta0=[0.4])
out_circle = odr_circle.run()
out_other = odr_other.run()
out_circle.pprint()
out_other.pprint()


xplot_circ = np.linspace(0.4, 1, 500)
xplot_other = np.linspace(0, 0.6, 500)
yplot_c0 = circle(out_circle.beta, xplot_circ)
yplot_sc = other_func(out_other.beta, xplot_other)

fit_label_circ = f'Fit: C = {out_circle.beta[0]:.2g} $\\pm$ {out_circle.sd_beta[0]:.2f} V/MHz'
fit_label_other = f'Fit: C = {out_other.beta[0]:.2g} $\\pm$ {out_other.sd_beta[0]:.2g} V/MHz'

plt.figure()
plt.errorbar(x_c0, D_values, yerr=D_unc, xerr=dx_c0,
             ls='', label='Data', color='blue', fmt='.')
plt.plot(xplot_circ, yplot_c0, label=fit_label_circ, color='red', linewidth=2)
plt.xlabel('Ac/A0')
plt.ylabel('D [V/MHz]')
plt.title(r'Power transfer: $\text{D}=\text{C}\sqrt{\frac{A_c}{A_0}\left(1-\frac{A_c}{A_0}\right)}$')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('data_non_confocal/figures/power_transfer_c0.png')

plt.figure()
plt.errorbar(x_sc, D_values, yerr=D_unc, xerr=dx_c0,
             ls='', label='Data', color='blue', fmt='.')
plt.plot(xplot_other, yplot_sc, label=fit_label_other, color='red', linewidth=2)
plt.xlabel('As/Ac')
plt.ylabel('D [V/MHz]')
plt.title(r'Power transfer: $\text{D}=\text{C}\sqrt{\frac{1}{1+2\frac{A_c}{A_0}}\left(1-\frac{1}{1+2\frac{A_c}{A_0}}\right)}$')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('data_non_confocal/figures/power_transfer_sc.png')
plt.show()
