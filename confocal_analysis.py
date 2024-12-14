import pandas as pd
import functions as fn
import fit_peaks as fp
import numpy as np
import os

R = 0.05  # m, mirror radius of curvature
c = 3e8  # speed of light
l = 50e-3  # cavity length (imposing)
fsr_freq = c/(2*l)  # (expected) fsr

#####################################################################################################
# section with file specific values

titles = ['scope_' + str(i) for i in range(15, 26)]
paths = [f'data_confocal/clean_data/{t}.csv' for t in titles]
os.makedirs(f"data_confocal/figures/calibration", exist_ok=True)

#####################################################################################################

with open('data_confocal/output.txt', 'a') as file:
    for path, title in zip(paths, titles):
        print(f'file: {title}', file=file)

        '''read data'''
        data = pd.read_csv(path)

        timestamps = data['timestamp'].to_numpy()
        volt_laser = data['volt_laser'].to_numpy()
        volt_piezo = data['volt_piezo'].to_numpy()

        '''find free spectral range (in voltage units) and finesse from peaks'''

        # find peaks and widths with just the find_peak function (only for plotting)
        xpeaks, ypeaks, peak_widths, indices = fn.peaks(
            volt_piezo, volt_laser, 0.08, 400, True)

        # find peaks and widths fitting lorentzians
        lor, cov = fp.fit_peaks(volt_piezo, volt_laser, 0.08, 400)

        # reoving unwanted peaks from data
        mask = ((xpeaks > -8.2) | (ypeaks > 0.6))
        xpeaks = xpeaks[mask]
        ypeaks = ypeaks[mask]
        peak_widths = peak_widths[mask]
        lor = np.array(lor)
        cov = np.array(cov)
        lor = lor[mask]
        cov = cov[mask]
        indices = indices[mask]

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

        print('Assuming confocality', file=file)

        # Generate expected frequencies
        expected_freq = np.arange(0, fsr_freq/2 * len(xpeaks), fsr_freq/2)

        # find calibration, and plot the data
        coeffs1, coeffs2, d_coeffs_1, d_coeffs_2 = fn.plot_fits(x0_list, expected_freq, "Peaks in piezo voltage (V)", "Expected frequency (Hz)",
                                                                f'data_confocal/figures/calibration/calibration_confocal_{title}.pdf',
                                                                title='Assuming confocality', save=True)

        print('linear coeffs (m * x + q)', file=file)
        print(f'm = {coeffs1[0]} +/- {d_coeffs_1[0]} Hz/V', file=file)
        print(f'q = {coeffs1[1]} +/- {d_coeffs_1[1]} Hz', file=file)
        print('quadratic coeffs (a * x^2 + b* x + c)', file=file)
        print(f'a = {coeffs2[0]} +/- {d_coeffs_2[0]} Hz/V^2', file=file)
        print(f'b = {coeffs2[1]} +/- {d_coeffs_2[1]} Hz/V', file=file)
        print(f'c = {coeffs2[2]} +/- {d_coeffs_2[2]} Hz', file=file)

        # convert piezo voltages into frequencies
        calibrated_freqs = coeffs2[0] * volt_piezo ** 2 + \
            coeffs2[1] * volt_piezo + coeffs2[2]

        fn.plot_piezo_laser_fit(volt_piezo, volt_laser, file_name=f'data_confocal/figures/calibration/piezo_peaks_{title}.png', A=A_list,
                         x0=x0_list, gamma=gamma_list, xpeaks=xpeaks, ypeaks=ypeaks, width=peak_widths, save=True)

        fn.scattering(x=calibrated_freqs, y=volt_laser, x_label='Relative frequency values (GHz)', y_label='Laser Intensity (V)',
                      title='Laser Intensity (calibrated): assuming confocality',
                      file_name=f'data_confocal/figures/calibration/data_confocal_{title}.png', save=True)

        print('Without assuming confocality', file=file)

        x0_nonconfoc = x0_list[::2]

        # generate expected frequencies
        expected_freq = np.arange(0, fsr_freq * len(x0_nonconfoc), fsr_freq)

        # find calibration, and plot the data
        coeffs1, coeffs2, _, _ = fn.plot_fits(x0_nonconfoc, expected_freq, "Peaks in piezo voltage (V)", "Expected frequency (Hz)",
                                              f'data_confocal/figures/calibration/calibration_non_confocal_{title}.pdf',
                                              title='Without assuming confocality', save=True)

        print('linear coeffs (m * x + q)', file=file)
        print(f'm = {coeffs1[0]} Hz/V', file=file)
        print(f'q = {coeffs1[1]} Hz', file=file)
        print('quadratic coeffs (a * x^2 + b* x + c)', file=file)
        print(f'a = {coeffs2[0]} Hz/V^2', file=file)
        print(f'b = {coeffs2[1]} Hz/V', file=file)
        print(f'c = {coeffs2[2]} Hz\n', file=file)

        # convert piezo voltages into frequencies
        calibrated_freqs_1 = coeffs2[0] * volt_piezo**2 + \
            coeffs2[1] * volt_piezo + coeffs2[2]

        fn.scattering(x=calibrated_freqs, y=volt_laser, x_label='Relative frequency values (GHz)', y_label='Laser Intensity (V)',
                      title='Laser Intensity (calibrated): without assuming confocality',
                      file_name=f'data_confocal/figures/calibration/data_non_confocal_{title}.png', save=True)

        data = {'freq_confoc': calibrated_freqs,
                'freq_non_confoc': calibrated_freqs_1,
                'volt_laser': volt_laser}
        df = pd.DataFrame(data)

        new_file_path = f"data_confocal/clean_data/{title}_calib.csv"
        df.to_csv(new_file_path, index=False)
        print(f"Data saved to {new_file_path}")

        peak_freq_conf = calibrated_freqs[indices]
        peak_freq_non_conf = calibrated_freqs_1[indices]
        time_peaks = timestamps[indices]

        pk = {'indices': indices,
              'timestamp': time_peaks,
              'laser_peaks': ypeaks,
              'piezo_peaks': xpeaks,
              'freq_conf': peak_freq_conf,
              'freq_non_confoc': peak_freq_non_conf,
              'lor_A': A_list,
              'lor_mean': x0_list,
              'lor_gamma': gamma_list,
              'lor_d_A': dA,
              'lor_d_mean': dx0,
              'lor_d_gamma': dgamma}

        df_pk = pd.DataFrame(pk)
        pk_file_path = f"data_confocal/clean_data/{title}_peaks.csv"
        df_pk.to_csv(pk_file_path, index=False)
        print(f"Peaks saved to {pk_file_path}")
