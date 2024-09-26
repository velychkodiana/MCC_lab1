import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# візьмемо із файлу (варіант 3)
data = np.loadtxt('/Users/macbookpro/Desktop/МСС/Lab1/f3.txt')

# ввід параметрів
T = 5
N = len(data)
delta_t = T / N
time_values = np.linspace(0, T, N)

# обчислення перетворення Фур'є
def calculate_dft(signal):
    N = len(signal)
    exp_factor = -2j * np.pi / N
    return np.array([np.sum(signal * np.exp(exp_factor * k * np.arange(N))) for k in range(N)]) / N

fourier_transform = calculate_dft(data)

# визначимо частоти
frequencies = np.fft.fftfreq(N, delta_t)
magnitude_fourier = np.abs(fourier_transform)

# визначимо піки
half_range = N // 2
peak_indices, _ = find_peaks(magnitude_fourier[:half_range])

# відбір значущих частот
min_threshold = 1
significant_frequencies = frequencies[peak_indices]
filtered_frequencies = significant_frequencies[np.abs(significant_frequencies) > min_threshold]

# метод найменших квадратів
def sine_wave(time, frequency):
    return np.sin(2 * np.pi * frequency * time)

def construct_matrix_and_vector(time, signal, freq):
    sine_t = sine_wave(time, freq[0])
    time_powers = [np.sum(time ** p) for p in range(7)]

    # матриця
    matrix = np.array([
        [time_powers[6], time_powers[5], time_powers[4], np.sum(sine_t * time ** 3), time_powers[3]],
        [time_powers[5], time_powers[4], time_powers[3], np.sum(sine_t * time ** 2), time_powers[2]],
        [time_powers[4], time_powers[3], time_powers[2], np.sum(sine_t * time), time_powers[1]],
        [np.sum(sine_t * time ** 3), np.sum(sine_t * time ** 2), np.sum(sine_t * time), np.sum(sine_t ** 2), np.sum(sine_t)],
        [time_powers[3], time_powers[2], time_powers[1], np.sum(sine_t), N]
    ])

    # вектор
    vector = np.array([
        np.sum(signal * time ** 3),
        np.sum(signal * time ** 2),
        np.sum(signal * time),
        np.sum(signal * sine_t),
        np.sum(signal)
    ])

    return matrix, vector

# вирішення системи рівнянь
def solve_least_squares(time, signal, freq):
    matrix, vector = construct_matrix_and_vector(time, signal, freq)
    return np.linalg.solve(matrix, vector)

parameters = solve_least_squares(time_values, data, filtered_frequencies)
rounded_parameters = np.round(parameters).astype(int)

# виведення результатів
print("важливі частоти:", filtered_frequencies)
print("підібрані параметри a:", rounded_parameters)

# рівняння моделі
def display_model_equation(params, freqs):
    equation = f"y(t) = {params[0]} * t^3 + {params[1]} * t^2 + {params[2]} * t + {params[3]} * sin(2π * {freqs[0]} * t) + {params[4]}"
    print("рівняння моделі:", equation)

display_model_equation(rounded_parameters, filtered_frequencies)

# побудова графіків
def plot_observations(time, signal):
    plt.figure(figsize=(10, 5))
    plt.plot(time, signal)
    plt.title('спостереження y(t) в залежності від часу')
    plt.xlabel(' час (в секундах)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.show()

def plot_fourier_transform(fregs, magnitudes):
    plt.figure()
    plt.plot(time_values, np.abs(fourier_transform))
    plt.title('модуль перетворення Фур\'є')
    plt.xlabel('частота')
    plt.ylabel('|c_y(k)|')
    plt.grid(True)
    plt.show()

def plot_peaks(freqs, magnitudes, peaks):
    plt.figure(figsize=(10, 5))
    plt.plot(freqs[:half_range], magnitudes[:half_range])
    plt.plot(freqs[peaks], magnitudes[peaks], 'x')
    plt.title('модуль перетворення Фур\'є з піками')
    plt.xlabel('частота')
    plt.ylabel('|c_y(k)|')
    plt.grid(True)
    plt.show()

def generate_model(params, time, freq):
    return (params[0] * time ** 3 + params[1] * time ** 2 + params[2] * time
            + params[3] * np.sin(2 * np.pi * freq[0] * time) + params[4])

def plot_model(time, model):
    plt.figure(figsize=(10, 5))
    plt.plot(time, model)
    plt.title('графік моделі з підібраними параметрами')
    plt.xlabel('час (в секундах)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.show()

# відображення графіків
plot_observations(time_values, data)
plot_fourier_transform(frequencies, magnitude_fourier)
plot_peaks(frequencies, magnitude_fourier, peak_indices)

fitted_model = generate_model(rounded_parameters, time_values, filtered_frequencies)
plot_model(time_values, fitted_model)

# середньоквадратична похибка
mse_value = np.mean((data - fitted_model) ** 2)
print(f"середньоквадратична похибка: {mse_value}")
