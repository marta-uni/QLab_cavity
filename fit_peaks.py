from scipy.signal import find_peaks, peak_widths
import numpy as np
from scipy.optimize import curve_fit


def lorentzian(x, A, x0, gamma):
    return A / (1 + ((x - x0) / gamma) ** 2)


def fit_peaks(x, y, height, distance):
    '''Given x and y (and peaks height and distance) returns a list of lorentzian curves
    (just the 3 parameters A, x0, gamma) and a list with the covariance matrices from the fit.'''

    x_spacing = np.mean(np.diff(x))

    # Find peaks and widths
    peaks, _ = find_peaks(y, height=height, distance=distance)
    widths_half = peak_widths(y, peaks, rel_height=0.5)[0]

    # Loop through each peak and fit
    params = []
    covs = []
    for peak, width in zip(peaks, widths_half):
        # Determine a fitting range around the peak, e.g., ±1.2 * width
        fit_range = int(width * 1.2)
        start = max(0, peak - fit_range)
        end = min(len(x), peak + fit_range)

        # Extract data around the peak
        x_fit_range = x[start:end]
        y_fit_range = y[start:end]
        width_scaled = width * x_spacing

        # Initial guess: A=height at peak, x0=peak position in x_fitted, gamma=half-width at half-maximum
        initial_guess = [y[peak], x[peak], width_scaled / 2]

        # Define bounds for A, x0, and gamma
        bounds = (
            # Lower bounds for [A, x0, gamma]
            [0, x[peak] - width_scaled, 0],
            # Upper bounds for [A, x0, gamma]
            [np.inf, x[peak] + width_scaled, width_scaled * 2]
        )

        try:
            popt, pcov = curve_fit(lorentzian, x_fit_range, y_fit_range,
                                   p0=initial_guess, bounds=bounds, maxfev=10000)
            params.append(popt)
            covs.append(pcov)
        except RuntimeError as e:
            print(
                f"Failed to fit peak at piezo_fitted = {x[peak]:.2f} due to RuntimeError: {e}")
        except Exception as e:
            print(
                f"An unexpected error occurred while fitting peak at piezo_fitted = {x[peak]:.2f}: {e}")

    return params, covs


def fit_peaks2(x, y, peaks, widths, ind_range):
    '''Given x and y, a list of peaks, FWHMs and the number of points to use in the fit, returns a list of lorentzian curves
    (just the 3 parameters A, x0, gamma) and a list with the covariance matrices from the fit.'''

    x_spacing = np.mean(np.diff(x))

    indices = [np.flatnonzero(x == pk)[0] for pk in peaks if pk in x]

    # Loop through each peak and fit
    params = []
    covs = []
    for peak_index, width in zip(indices, widths):
        # Determine a fitting range around the peak, e.g., ±1.2 * width
        start = max(0, peak_index - int(ind_range/2))
        end = min(len(x), peak_index + int(ind_range/2))

        # Extract data around the peak
        x_fit_range = x[start:end]
        y_fit_range = y[start:end]

        # Initial guess: A=height at peak, x0=peak position in x_fitted, gamma=half-width at half-maximum
        initial_guess = [y[peak_index], x[peak_index], width / 2]

        # Define bounds for A, x0, and gamma
        bounds = (
            # Lower bounds for [A, x0, gamma]
            [0, x[peak_index] - width, 0],
            # Upper bounds for [A, x0, gamma]
            [np.inf, x[peak_index] + width, width * 2]
        )

        try:
            popt, pcov = curve_fit(lorentzian, x_fit_range, y_fit_range,
                                   p0=initial_guess, bounds=bounds, maxfev=10000)
            params.append(popt)
            covs.append(pcov)
        except RuntimeError as e:
            print(
                f"Failed to fit peak at piezo_fitted = {x[peak_index]:.2f} due to RuntimeError: {e}")
        except Exception as e:
            print(
                f"An unexpected error occurred while fitting peak at piezo_fitted = {x[peak_index]:.2f}: {e}")

    return params, covs


def fit_peaks3(x, y, peaks, widths):
    '''Given x and y, a list of peaks and FWHMs, returns a list of lorentzian curves
    (just the 3 parameters A, x0, gamma) and a list with the covariance matrices from the fit.'''

    x_spacing = np.mean(np.diff(x))

    indices = [np.flatnonzero(x == pk)[0] for pk in peaks if pk in x]

    # Loop through each peak and fit
    params = []
    covs = []
    for peak_index, width in zip(indices, widths):
        # Determine a fitting range around the peak, e.g., ±1.2 * width
        start = max(0, peak_index - int(width / (x_spacing * 2)))
        end = min(len(x), peak_index + int(width / (x_spacing * 2)))

        # Extract data around the peak
        x_fit_range = x[start:end]
        y_fit_range = y[start:end]

        # Initial guess: A=height at peak, x0=peak position in x_fitted, gamma=half-width at half-maximum
        initial_guess = [y[peak_index], x[peak_index], width / 2]

        # Define bounds for A, x0, and gamma
        bounds = (
            # Lower bounds for [A, x0, gamma]
            [0, x[peak_index] - width, 0],
            # Upper bounds for [A, x0, gamma]
            [np.inf, x[peak_index] + width, width * 2]
        )

        try:
            popt, pcov = curve_fit(lorentzian, x_fit_range, y_fit_range,
                                   p0=initial_guess, bounds=bounds, maxfev=10000)
            params.append(popt)
            covs.append(pcov)
        except RuntimeError as e:
            print(
                f"Failed to fit peak at piezo_fitted = {x[peak_index]:.2f} due to RuntimeError: {e}")
        except Exception as e:
            print(
                f"An unexpected error occurred while fitting peak at piezo_fitted = {x[peak_index]:.2f}: {e}")

    return params, covs

import matplotlib.pyplot as plt
from matplotlib import cm
def plot_piezo_laser_fit2(piezo_fitted, volt_laser, file_name, A, x0, gamma, xpeaks, ypeaks, ind_range, save=False):
    fitted_curves = []
    piezo_spacing = np.mean(np.diff(piezo_fitted))
    for A_, x0_, gamma_, peak in zip(A, x0, gamma, xpeaks):
        x = np.linspace(peak - ind_range * piezo_spacing/2, peak + ind_range * piezo_spacing/2, 100)
        y = lorentzian(x, A_, x0_, gamma_)
        fitted_curves.append((x, y))

    cmap = cm.get_cmap('Oranges')
    colors = cmap(np.linspace(0.5, 0.9, len(fitted_curves)))

    plt.figure(figsize=(12, 6))
    plt.scatter(piezo_fitted, volt_laser, label='Laser Intensity vs. Piezo volt',
                color='green', marker='.')
    plt.scatter(xpeaks, ypeaks, marker='x', label='Peak Values')
    for i, (x, y) in enumerate(fitted_curves):
        plt.plot(x, y, '--', label=f'Fitted Lorentzian {i+1}', color=colors[i])
    plt.xlabel('Voltage Piezo (V)')
    plt.ylabel('Laser Intensity (V)')
    plt.title('Piezo Voltage vs Laser Voltage')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save:
        plt.savefig(file_name)
        plt.close()
    else:
        plt.show()