import pandas as pd
import functions as fn
import fit_peaks as fp
import numpy as np
import os

#####################################################################################################
# section with file specific values

title_list = [f'error{str(i).zfill(5)}' for i in range(1, 19)]
# excluding files with "throw out note" (8 to 10), files with unclear sidebands (5 to 7 and 16),
# and files i don't understand (0, 18)
title_list = title_list[:4] + title_list[10:15] + [title_list[16]]
# after changing the current the peak threshold got higher
min_heights = [0.02, 0.02, 0.02, 0.02,
               0.045, 0.045, 0.045, 0.045, 0.045, 0.04]
# fsr from first analysis
fsr_volt = 1.5596783075517309
d_fsr_volt = 0.0001395265273752087

os.makedirs(f"data_non_confocal/figures/finesse", exist_ok=True)

ind_range = 400

#####################################################################################################

finesse_list = []
d_finesse_list = []

with open("data_non_confocal/finesse_double_fit.txt", "w") as file:
    for title, h in zip(title_list, min_heights):
        data = pd.read_csv(f'data_non_confocal/clean_data/{title}.csv')
        volt_piezo = data['volt_piezo'].to_numpy()
        volt_laser = data['transmission'].to_numpy()

        file.write(f'\nPeaks in {title}\n')

        xpeaks, ypeaks, peak_widths = fn.peaks(volt_piezo, volt_laser, h, 400)
        peak_widths = [max(pw, 0.001) for pw in peak_widths]
        lor, cov = fp.fit_peaks2(
            volt_piezo, volt_laser, xpeaks, peak_widths, ind_range)

        new_widths = [(2 * popt[2]) for popt in lor]

        lor, cov = fp.fit_peaks3(volt_piezo, volt_laser, xpeaks, new_widths)

        x0_list = []
        A_list = []
        gamma_list = []
        dA = []
        dx0 = []
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
        dA = np.array(dA)
        dx0 = np.array(dx0)
        dgamma = np.array(dgamma)

        for a_, da_, x0_, dx0_, g_, dg_ in zip(A_list, dA, x0_list, dx0, gamma_list, dgamma):
            file.write(f'A = {a_} +/- {da_} V\n')
            file.write(f'x0 = {x0_} +/- {dx0_} V\n')
            file.write(f'gamma = {g_} +/- {dg_} V\n')

        fp.plot_piezo_laser_fit2(volt_piezo, volt_laser, file_name=f'data_non_confocal/figures/finesse/piezo_peaks_double_fit_{title}.png', A=A_list,
                                 x0=x0_list, gamma=gamma_list, xpeaks=xpeaks, ypeaks=ypeaks, ind_range=ind_range, save=True)

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
