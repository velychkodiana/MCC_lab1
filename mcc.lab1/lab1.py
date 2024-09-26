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

# зобразимо перетворення Фур'є
def calculate_dft(signal):
    N = len(signal)
    result = np.zeros(N, dtype=complex)
    for k in range(N):
        for m in range(N):
            result[k] += signal[m] * np.exp(-2j * np.pi * k * m / N)
        result[k] /= N  # нормалізація
    return result

fourier_transform = calculate_dft(data)

# визначимо частоти
frequencies = np.fft.fftfreq(N, delta_t)
magnitude_fourier = np.abs(fourier_transform)

# визначимо піки 
half_range = N // 2
peak_indices, _ = find_peaks(magnitude_fourier[:half_range])

# оберемо найбільш значущі частоти
significant_frequencies = frequencies[peak_indices]
min_threshold = 1
filtered_frequencies = significant_frequencies[np.abs(significant_frequencies) > min_threshold]

# використаємо метод найменших квадратів
def sine_wave(time, frequency):
    return np.sin(2 * np.pi * frequency * time)

# побудуємо матрицю для системи рівнянь
def construct_matrix(time, freq):
    matrix = np.zeros((5, 5))

    matrix[0, 0] = np.sum(time ** 6)
    matrix[0, 1] = np.sum(time ** 5)
    matrix[0, 2] = np.sum(time ** 4)
    matrix[0, 3] = np.sum(sine_wave(time, freq[0]) * time ** 3)
    matrix[0, 4] = np.sum(time ** 3)

    matrix[1, 0] = np.sum(time ** 5)
    matrix[1, 1] = np.sum(time ** 4)
    matrix[1, 2] = np.sum(time ** 3)
    matrix[1, 3] = np.sum(sine_wave(time, freq[0]) * time ** 2)
    matrix[1, 4] = np.sum(time ** 2)

    matrix[2, 0] = np.sum(time ** 4)
    matrix[2, 1] = np.sum(time ** 3)
    matrix[2, 2] = np.sum(time ** 2)
    matrix[2, 3] = np.sum(sine_wave(time, freq[0]) * time)
    matrix[2, 4] = np.sum(time)

    matrix[3, 0] = np.sum(sine_wave(time, freq[0]) * time ** 3)
    matrix[3, 1] = np.sum(sine_wave(time, freq[0]) * time ** 2)
    matrix[3, 2] = np.sum(sine_wave(time, freq[0]) * time)
    matrix[3, 3] = np.sum(sine_wave(time, freq[0]) ** 2)
    matrix[3, 4] = np.sum(N * sine_wave(time, freq[0]))

    matrix[4, 0] = np.sum(time ** 3)
    matrix[4, 1] = np.sum(time ** 2)
    matrix[4, 2] = np.sum(time)
    matrix[4, 3] = np.sum(N * sine_wave(time, freq[0]))
    matrix[4, 4] = N

    return matrix

# виведемо вектор результатів для системи рівнянь
def construct_vector(time, signal, freq):
    vector = np.array([
        np.sum(signal * time ** 3),
        np.sum(signal * time ** 2),
        np.sum(signal * time),
        np.sum(signal * sine_wave(time, freq[0])),
        np.sum(signal)
    ])
    return vector

# знайдемо розв'язок (метод найменших квадратів)
def solve_least_squares(time, signal, freq):
    A = construct_matrix(time, freq)
    c = construct_vector(time, signal, freq)
    return np.linalg.solve(A, c)

parameters = solve_least_squares(time_values, data, filtered_frequencies)
rounded_parameters = np.round(parameters).astype(int)

print("важливі частоти:", filtered_frequencies)
print("підібрані параметри a:", rounded_parameters)

# сторимо рівняння для моделі
def display_model_equation(params, freqs):
    equation = f"y(t) = {params[0]} * t^3 + {params[1]} * t^2 + {params[2]} * t + {params[3]} * sin(2π * {freqs[0]} * t) + {params[4]}"
    print("рівняння моделі:", equation)

display_model_equation(rounded_parameters, filtered_frequencies)

# зобразимо графік спостережень
plt.figure(figsize=(10, 5))
plt.plot(time_values, data)
plt.title('спостереження y(t) в залежності від часу')
plt.xlabel(' час (в секундах)')
plt.ylabel('y(t)')
plt.grid(True)
plt.show()

# зобразимо графік модуля перетворення Фур'є
plt.figure(figsize=(10, 5))
plt.plot(frequencies[:N], magnitude_fourier[:N])
plt.title('модуль перетворення Фур\'є')
plt.xlabel('частота')
plt.ylabel('|c_y(k)|')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(frequencies[:half_range], magnitude_fourier[:half_range])
plt.plot(frequencies[peak_indices], magnitude_fourier[peak_indices], 'x')
plt.title('модуль перетворення Фур\'є з піками')
plt.xlabel('частота')
plt.ylabel('|c_y(k)|')
plt.grid(True)
plt.show()

# нарешті!!! зобразимо модель з підібраними параметрами
def generate_model(params, time, freq):
    return params[0] * time ** 3 + params[1] * time ** 2 + params[2] * time + params[3] * np.sin(2 * np.pi * freq[0] * time) + params[4]

fitted_model = generate_model(rounded_parameters, time_values, filtered_frequencies)

plt.figure(figsize=(10, 5))
plt.plot(time_values, fitted_model)
plt.title('графік моделі з підібраними параметрами')
plt.xlabel('час (в секундах)')
plt.ylabel('y(t)')
plt.grid(True)
plt.show()

# виведемо середньоквадратичну похибку
mse_value = np.mean((data - fitted_model) ** 2)
print(f"середньоквадратична похибка: {mse_value}")
