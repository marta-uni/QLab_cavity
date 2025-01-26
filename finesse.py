import pandas as pd
import functions as fn
import fit_peaks as fp
import numpy as np
import os

#####################################################################################################
# section with file specific values

title_list = [f'error{str(i).zfill(5)}' for i in range(1, 19)]
# excluding files with "throw out note" (8 to 10), files with unclear sidebands (5 to 7 and 16),
# and files i don't understand (0, 18), files 3 and 4 aren't considered because of a threshold we put at 40 MHz
# i don't know about 17
title_list = title_list[:2] + title_list[10:15]
# after changing the current the peak threshold got higher
min_heights = [0.02, 0.02, 0.045, 0.045, 0.045, 0.045, 0.045]
# fsr from first analysis
fsr_volt = 1.5596783075517309
d_fsr_volt = 0.0001395265273752087

os.makedirs(f"data_non_confocal/figures/finesse", exist_ok=True)
os.makedirs(f"data_non_confocal/figures/finesse/residuals", exist_ok=True)

ind_range = 200

write_file_name = "data_non_confocal/finesse_leonardi.txt"
figure_name = 'peaks_leonardi'
# order to change from double fit to single fit you have to modify these 2 titles
# and use fit_peaks2 for single fit and fit_peaks3 for double fit

#####################################################################################################

finesse_list = []
d_finesse_list = []

with open(write_file_name, "w") as file:
    for title, h in zip(title_list, min_heights):
        data = pd.read_csv(f'data_non_confocal/clean_data/{title}.csv')
        volt_piezo = data['volt_piezo'].to_numpy()
        volt_laser = data['transmission'].to_numpy()

        file.write(f'\nPeaks in {title}\n')

        xpeaks, ypeaks, peak_widths = fn.peaks(volt_piezo, volt_laser, h, 400)
        peak_widths = [max(pw, 0.001) for pw in peak_widths]
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

        indices = [np.flatnonzero(volt_piezo == pk)[0]
                   for pk in xpeaks if pk in volt_piezo]

        for popt, pcov, i in zip(lor, cov, indices):
            A_list.append(popt[0])
            x0_list.append(popt[1])
            gamma_list.append(popt[2])
            off_list.append(popt[3])
            dA.append(np.sqrt(pcov[0, 0]))
            dx0.append(np.sqrt(pcov[1, 1]))
            dgamma.append(np.sqrt(pcov[2, 2]))
            doff.append(np.sqrt(pcov[3, 3]))
            fn.fit_residuals(fp.lorentzian_off, volt_piezo[i-ind_range//2:i+ind_range//2], volt_laser[i-ind_range//2:i+ind_range//2], popt, 'Volt piezo [V]',
                             'Laser intensity [V]', f'Residuals for peak in {volt_piezo[i]:.3g} V', f'data_non_confocal/figures/finesse/residuals/{figure_name}_{title}_{volt_piezo[i]:.3g}.png', True)

        x0_list = np.array(x0_list)
        A_list = np.array(A_list)
        gamma_list = np.array(gamma_list)
        off_list = np.array(off_list)
        dA = np.array(dA)
        dx0 = np.array(dx0)
        dgamma = np.array(dgamma)
        doff = np.array(doff)

        for a_, da_, x0_, dx0_, g_, dg_, o_, do_ in zip(A_list, dA, x0_list, dx0, gamma_list, dgamma, off_list, doff):
            file.write(f'A = {a_} +/- {da_} V\n')
            file.write(f'x0 = {x0_} +/- {dx0_} V\n')
            file.write(f'gamma = {g_} +/- {dg_} V\n')
            file.write(f'off = {o_} +/- {do_} V\n\n')

        fp.plot_piezo_laser_fit_leonardi(volt_piezo, volt_laser, file_name=f'data_non_confocal/figures/finesse/{figure_name}_{title}.png', A=A_list,
                                         x0=x0_list, gamma=gamma_list, off=off_list, xpeaks=xpeaks, ypeaks=ypeaks, ind_range=ind_range, save=True)

        fin = fsr_volt/(2*gamma_list)
        d_fin = fin * np.sqrt((d_fsr_volt / fsr_volt) **
                              2 + (dgamma / gamma_list)**2)

        d_finesse_list.extend(d_fin)
        finesse_list.extend(fin)

    finesse_list = np.array(finesse_list)
    d_finesse_list = np.array(d_finesse_list)

    weights = 1 / d_finesse_list**2

    finesse = np.sum(weights * finesse_list) / np.sum(weights)
    d_finesse = np.sqrt(1 / np.sum(weights))

    file.write(f'\nFinesse = {finesse} +/- {d_finesse}')
