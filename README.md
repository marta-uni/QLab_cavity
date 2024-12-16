# QLab_cavity

Tasks:
- find piezo calibration
- calculate cavity finesse
- estimate fsr and L (if non confocal)
- estimate errors on calibration

data_confocal:
We took this data on 12.11.24, data collections span two fsr in (almost) confocal configuration. Laser temperature was 23.790°C. scope_15 to scope_20 had 49.484 mA of laser current, piezo range was 15 Vpp, with offset -2.5V. scope_21 had 48.951 mA of laser current, piezo range was 15 Vpp, with offset -2.5V. scope_22 to scope_25 had 49.484 mA of laser current, piezo range was 15 Vpp, with offset -1V.
- First step is to remove NaNs and crop data to include only a single frequency sweep from the piezo, this is performed in data_prep.py. Data to analyse is saved in data_confocal/clean_data.
- Then we extract peaks information by fitting them to a lorenztian. Fit results (and all the informations concerning the peaks) are saved, for each scope_X file in data_confocal/clean_data/scope_X.csv.
- We calibrate the spectrum (at least we for what concerns frequency distances) imposing given frequencies for each peak and fitting piezo volatge corresponding to the peak to a linear and a quadratic frequency conversion. We find that quadratic conversion is more accurare than the linear one. We repeat the process by considering only the 3 TEM00 points. We save laser voltage and calibrated frequencies (for both calibrations) in scope_X_calib.csv.
- The last 2 steps are done in confocal_analysis.py, lorentzian fitting functions are in fit_peaks.py, other useful functions can be found in functions.py.
- We then check the accuracy of the confocality assumption in two ways
    - We expect the odd modes peaks to be extactly at half fsr from TEM00, so we measure the displacement of these odd modes from the expected frequency. We find that separation is of the order ~10MHz.
    - We expect all the peaks to collapse into the TEM00 mode and the TEM01/TEM10 mode, however we can observe small structures around the odd modes. We estimate the separation of the odd modes from the nearest small peaks to be in the order ~10MHz.
  These calculations are done in displacement.py

From non confocal
- find finesse ( HWHM from zoomed file with sidebands and FSR from zoomed out file )
- piezo calibration

Tasks 2
- Variazione di frequenza
  - calcolare pendenza segnale errore e vedere se è quella che ci aspettiamo dal calcolo teorico
  - NB usare le sidebands per calibrare
- Variazione di ampiezza a freq fissata (corrente nel diodo era 20.01, non modulata)
  - guardare quanta potenza passa dal picco principale alle sidebands al variare dell'ampiezza (funzioni di Bessel? what?)
  - roba del semicerchio
- Studiare asimmetria sidebands se vogliamo
