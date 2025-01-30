import pandas as pd
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
import numpy as np
from scipy.special import jv
from scipy.optimize import fsolve, curve_fit

df = pd.read_csv('data_bessel/clean_data/bessel_sidebands.csv')
df = df[df['title'] != 'bessel00003']
df = df[df['title'] != 'bessel00004']

print(df)

grouped_data = {label: group for label, group in df.groupby(df['title'])}

'''mod_ampl = np.array([10, 50, 110, 140, 170, 180, 190, 200, 220,
                     250, 270, 290, 310, 330, 350, 370, 390, 410, 430, 450])/1000'''

mod_ampl = np.array([10, 50, 110, 180, 190, 200, 220,
                     250, 270, 290, 310, 330, 350, 370, 390, 410, 430, 450])/1000

A_s1 = []
dA_s1 = []
mod_carr = []
A_s2 = []
dA_s2 = []
mod_sb1 = []
A_c = []
dA_c = []
mod_sb2 = []

for i, (label, data) in enumerate(grouped_data.items()):
    total_sum = sum(unp.uarray(
        data['lor_A'].to_numpy(), data['lor_d_A'].to_numpy()))

    pk_types = {label: group for label, group in data.groupby(data['type'])}

    for pk, data2 in pk_types.items():
        mean_val = sum(unp.uarray(data2['lor_A'].to_numpy(
        ), data2['lor_d_A'].to_numpy())) / len(data2.index)
        norm_val = mean_val / total_sum
        if pk == 'carrier':
            A_c.append(norm_val.nominal_value)
            dA_c.append(norm_val.std_dev)
            mod_carr.append(mod_ampl[i])
        elif pk == 'sb_1':
            A_s1.append(norm_val.nominal_value)
            dA_s1.append(norm_val.std_dev)
            mod_sb1.append(mod_ampl[i])
        else:
            A_s2.append(norm_val.nominal_value)
            dA_s2.append(norm_val.std_dev)
            mod_sb2.append(mod_ampl[i])


mod_carr = np.array(mod_carr)
mod_sb1 = np.array(mod_sb1)
mod_sb2 = np.array(mod_sb2)


def int_01(x):
    return (jv(1, x)) ** 2 - (jv(0, x)) ** 2


def int_02(x):
    return (jv(2, x)) ** 2 - (jv(0, x)) ** 2


def linear(x, a, b):
    return a * x + b


intersection_1 = fsolve(int_01, 1.5)[0]
intersection_2 = fsolve(int_02, 2)[0]

print(intersection_1, intersection_2)

x_fit = [0, mod_sb2[1], mod_sb2[-1]]
y_fit = [0, intersection_1, intersection_2]

popt, pcov = curve_fit(linear, x_fit, y_fit)

conversion = popt[0]
d_conv = np.sqrt(pcov[0,0])
print(f'V_pi = {1/conversion/np.pi} +/- {d_conv/((conversion**2)*np.pi)} V')

x = np.linspace(mod_ampl[0], mod_ampl[-1], 100) * conversion

plt.figure()
plt.errorbar(mod_carr * conversion, A_c, dA_c, ls='',
             label='Carrier', color='blue', fmt='.')
plt.errorbar(mod_sb1 * conversion, A_s1, dA_s1, ls='',
             label='Sideband 1', color='red', fmt='.')
plt.errorbar(mod_sb2 * conversion, A_s2, dA_s2, ls='',
             label='Sideband 2', color='green', fmt='.')
plt.plot(x, (jv(0, x))**2,
         label='$(J_0)^2$', color='purple', linewidth=2)
plt.plot(x, (jv(1, x))**2,
         label='$(J_1)^2$', color='brown', linewidth=2)
plt.plot(x, (jv(2, x))**2,
         label='$(J_2)^2$', color='black', linewidth=2)
plt.xlabel(r'$\beta$')
plt.ylabel('Power on each component')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('data_bessel/figures/bessel_functions.png')
plt.show()
