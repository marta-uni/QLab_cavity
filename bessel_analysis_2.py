import pandas as pd
import matplotlib.pyplot as plt
from uncertainties import ufloat
import uncertainties.unumpy as unp
import numpy as np
from scipy.special import jv
from scipy.optimize import curve_fit
from scipy import odr

df = pd.read_csv('data_bessel/clean_data/bessel_sidebands.csv')
df = df[df['title'] != 'bessel00003']
df = df[df['title'] != 'bessel00004']

grouped_data = {label: group for label, group in df.groupby(df['title'])}

'''mod_ampl = np.array([10, 50, 110, 140, 170, 180, 190, 200, 220,
                     250, 270, 290, 310, 330, 350, 370, 390, 410, 430, 450])/1000'''

mod_ampl = np.array([10, 50, 110, 180, 190, 200, 220,
                     250, 270, 290, 310, 330, 350, 370, 390, 410, 430, 450])/1000

A_s1_l = []
dA_s1_l = []
mod_sb1_l = []
A_s2_l = []
dA_s2_l = []
mod_sb2_l = []

A_s1_r = []
dA_s1_r = []
mod_sb1_r = []
A_s2_r = []
dA_s2_r = []
mod_sb2_r = []

A_s1_mean = []
dA_s1_mean = []
mod_sb1_mean = []
A_s2_mean = []
dA_s2_mean = []
mod_sb2_mean = []

A_c = []
dA_c = []
mod_carr = []

for i, (label, data) in enumerate(grouped_data.items()):
    total_sum = sum(unp.uarray(
        data['lor_A'].to_numpy(), data['lor_d_A'].to_numpy()))

    pk_types = {label: group for label, group in data.groupby(data['type'])}

    for pk, data2 in pk_types.items():
        mean_val = sum(unp.uarray(data2['lor_A'].to_numpy(
        ), data2['lor_d_A'].to_numpy())) / len(data2.index)
        norm_val_mean = mean_val / total_sum
        if pk == 'carrier':
            A_c.append(norm_val_mean.nominal_value)
            dA_c.append(norm_val_mean.std_dev)
            mod_carr.append(mod_ampl[i])
        elif pk == 'sb_1':
            val_r = unp.uarray(data2['lor_A'].to_numpy(),
                               data2['lor_d_A'].to_numpy())[0]
            norm_val_r = val_r / total_sum
            A_s1_r.append(norm_val_r.nominal_value)
            dA_s1_r.append(norm_val_r.std_dev)
            mod_sb1_r.append(mod_ampl[i])
            A_s1_mean.append(norm_val_mean.nominal_value)
            dA_s1_mean.append(norm_val_mean.std_dev)
            mod_sb1_mean.append(mod_ampl[i])
            if len(data2.index) == 2:
                val_l = unp.uarray(
                    data2['lor_A'].to_numpy(), data2['lor_d_A'].to_numpy())[1]
                norm_val_l = val_l / total_sum
                A_s1_l.append(norm_val_l.nominal_value)
                dA_s1_l.append(norm_val_l.std_dev)
                mod_sb1_l.append(mod_ampl[i])
        else:
            val_r = unp.uarray(data2['lor_A'].to_numpy(),
                               data2['lor_d_A'].to_numpy())[0]
            norm_val_r = val_r / total_sum
            A_s2_r.append(norm_val_r.nominal_value)
            dA_s2_r.append(norm_val_r.std_dev)
            mod_sb2_r.append(mod_ampl[i])
            A_s2_mean.append(norm_val_mean.nominal_value)
            dA_s2_mean.append(norm_val_mean.std_dev)
            mod_sb2_mean.append(mod_ampl[i])
            val_l = unp.uarray(data2['lor_A'].to_numpy(),
                               data2['lor_d_A'].to_numpy())[1]
            norm_val_l = val_l / total_sum
            A_s2_l.append(norm_val_l.nominal_value)
            dA_s2_l.append(norm_val_l.std_dev)
            mod_sb2_l.append(mod_ampl[i])


mod_carr = np.array(mod_carr)
mod_sb1_l = np.array(mod_sb1_l)
mod_sb1_r = np.array(mod_sb1_r)
mod_sb2_l = np.array(mod_sb2_l)
mod_sb2_r = np.array(mod_sb2_r)
mod_sb1_mean = np.array(mod_sb1_mean)
mod_sb2_mean = np.array(mod_sb2_mean)


def bessel0_squared(x, k):
    return (jv(0, k * x))**2


popt_beta, pcov_beta = curve_fit(
    bessel0_squared, mod_carr, A_c, sigma=dA_c, absolute_sigma=True, p0=[4], bounds=(0, np.inf))

x_first_plot = np.linspace(min(mod_carr), max(mod_carr), 100)
y_first_plot = bessel0_squared(x_first_plot, *popt_beta)

k = ufloat(popt_beta[0], np.sqrt(pcov_beta[0, 0]))
v_pi = np.pi / k

plt.figure()
plt.errorbar(mod_carr, A_c, dA_c, ls='',
             label='Carrier', color='blue', fmt='.')
plt.plot(x_first_plot, y_first_plot, label='$(J_0)^2$',
         color='red', linewidth=2)
plt.xlabel(r'$V_{pp}$')
plt.ylabel('Normalized power')
plt.title(f'Estimating $V_\\pi =$ ({v_pi}) V')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('data_bessel/figures/bessel_0.png')

beta_carr = mod_carr * k
beta_sb1_mean = mod_sb1_mean * k
beta_sb2_mean = mod_sb2_mean * k
beta_sb1_l = mod_sb1_l * k
beta_sb1_r = mod_sb1_r * k
beta_sb2_l = mod_sb2_l * k
beta_sb2_r = mod_sb2_r * k

x_second_plot = np.linspace(min(unp.nominal_values(
    beta_carr)), max(unp.nominal_values(beta_carr)), 100)
y_second_plot_0 = (jv(0, x_second_plot))**2
y_second_plot_1 = (jv(1, x_second_plot))**2
y_second_plot_2 = (jv(2, x_second_plot))**2

plt.figure()
plt.errorbar(unp.nominal_values(beta_carr), A_c, xerr=unp.std_devs(beta_carr), yerr=dA_c, ls='',
             label='Carrier', color='blue', fmt='.')
plt.errorbar(unp.nominal_values(beta_sb1_mean), A_s1_mean, xerr=unp.std_devs(beta_sb1_mean), yerr=dA_s1_mean, ls='',
             label='Sideband 1', color='red', fmt='.')
plt.errorbar(unp.nominal_values(beta_sb2_mean), A_s2_mean, xerr=unp.std_devs(beta_sb2_mean), yerr=dA_s2_mean, ls='',
             label='Sideband 2', fmt='.g')
plt.plot(x_second_plot, y_second_plot_0,
         label='$(J_0)^2$', color='blue', linewidth=2)
plt.plot(x_second_plot, y_second_plot_1,
         label='$(J_1)^2$', color='red', linewidth=2)
plt.plot(x_second_plot, y_second_plot_2,
         label='$(J_2)^2$', color='green', linewidth=2)
plt.xlabel(r'$\beta$')
plt.ylabel('Power on each component')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('data_bessel/figures/bessel_functions_mean.png')


def sb1l_func(par, x):
    return (-jv(1, x) + par[0]/2*(jv(0, x) + jv(2, x)))**2


def sb1r_func(par, x):
    return (jv(1, x) + par[0]/2*(jv(0, x) + jv(2, x)))**2


def sb2l_func(par, x):
    return (jv(2, x) - par[0]/2*(jv(1, x) + jv(3, x)))**2


def sb2r_func(par, x):
    return (jv(2, x) + par[0]/2*(jv(1, x) + jv(3, x)))**2


def odr_fit(func, x, y, xerr, yerr, beta0):
    mymodel = odr.Model(func)
    data = odr.RealData(x, y, xerr, yerr)
    odr_ = odr.ODR(data, mymodel, beta0=beta0)
    out_ = odr_.run()
    return out_


out_sb1l = odr_fit(sb1l_func, unp.nominal_values(beta_sb1_l),
                   A_s1_l, unp.std_devs(beta_sb1_l), dA_s1_l, [0.1])
out_sb1r = odr_fit(sb1r_func, unp.nominal_values(beta_sb1_r),
                   A_s1_r, unp.std_devs(beta_sb1_r), dA_s1_r, [0.1])
out_sb2l = odr_fit(sb2l_func, unp.nominal_values(beta_sb2_l),
                   A_s2_l, unp.std_devs(beta_sb2_l), dA_s2_l, [0.1])
out_sb2r = odr_fit(sb2r_func, unp.nominal_values(beta_sb2_r),
                   A_s2_r, unp.std_devs(beta_sb2_r), dA_s2_r, [0.1])

y_third_plot_0 = sb1l_func(out_sb1l.beta, x_second_plot)
y_third_plot_1 = sb1r_func(out_sb1r.beta, x_second_plot)
y_third_plot_2 = sb2l_func(out_sb2l.beta, x_second_plot)
y_third_plot_3 = sb2r_func(out_sb2r.beta, x_second_plot)

plt.figure()
plt.errorbar(unp.nominal_values(beta_sb1_l), A_s1_l, xerr=unp.std_devs(beta_sb1_l), yerr=dA_s1_l, ls='',
             label='SB1 left', color='blue', fmt='.')
plt.errorbar(unp.nominal_values(beta_sb1_r), A_s1_r, xerr=unp.std_devs(beta_sb1_r), yerr=dA_s1_r, ls='',
             label='SB1 right', color='red', fmt='.')
plt.errorbar(unp.nominal_values(beta_sb2_l), A_s2_l, xerr=unp.std_devs(beta_sb2_l), yerr=dA_s2_l, ls='',
             label='SB2 left', color='green', fmt='.')
plt.errorbar(unp.nominal_values(beta_sb2_r), A_s2_r, xerr=unp.std_devs(beta_sb2_r), yerr=dA_s2_r, ls='',
             label='SB2 right', color='orange', fmt='.')
plt.plot(x_second_plot, y_third_plot_0,
         label=f'$\\alpha = $ {out_sb1l.beta[0]:.2g} $\\pm$ {out_sb1l.sd_beta[0]:.2g}', color='blue', linewidth=2)
plt.plot(x_second_plot, y_third_plot_1,
         label=f'$\\alpha = $ {out_sb1r.beta[0]:.2g} $\\pm$ {out_sb1r.sd_beta[0]:.2g}', color='red', linewidth=2)
plt.plot(x_second_plot, y_third_plot_2,
         label=f'$\\alpha = $ {out_sb2l.beta[0]:.2g} $\\pm$ {out_sb2l.sd_beta[0]:.2g}', color='green', linewidth=2)
plt.plot(x_second_plot, y_third_plot_3,
         label=f'$\\alpha = $ {out_sb2r.beta[0]:.2g} $\\pm$ {out_sb2r.sd_beta[0]:.2g}', color='orange', linewidth=2)
plt.xlabel(r'$\beta$')
plt.ylabel('Power on each component')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('data_bessel/figures/bessel_functions_left_right.png')
plt.show()