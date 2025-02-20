import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import json
import os

b = 1.0

r_range = [10, 20, 40, 60, 80, 120, 150, 180, 1000]

def load_csv(file_path):
    data = pd.read_csv(file_path, header=None)
    # data = pd.read_csv(file_path, header=None, skiprows=1, usecols=range(1, 42))
    
    # Check if data is 41x41 after skipping the first row and column
    if data.shape != (41, 41):
        raise ValueError("CSV file does not contain 40x40 data after skipping the first row and column")
    
    # for i in range(data.shape[0]):
    #     for j in range(data.shape[1]):
    #         if pd.isna(data.iat[i, j]):
    #             data.iat[i, j] = 0.0

    return data


def plot_heat_map_alone(data):
    plt.imshow(data, cmap='jet', interpolation='nearest')
    plt.colorbar()
    plt.show()

def plot_heat_map(data, ax, title):
    heatmap = ax.imshow(data, cmap='jet', interpolation='nearest', extent=[-200, 200, -200, 200])
    plt.colorbar(heatmap, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')


def calculate_distance_from_center(i, j, center=(20, 20), grid_size=10, a=1.0):
    x = (j - center[1]) * grid_size
    y = (i - center[0]) * grid_size

    distance = np.sqrt((x/a)**2 + (y/b)**2)
    return distance


def calculate_distances(data, center=(20, 20), grid_size=10):
    v_range = [0 for _ in r_range]
    v_num = [0 for _ in r_range]

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if pd.isna(data.iat[i, j]):
                continue
            distance = calculate_distance_from_center(i, j, center, grid_size)
            for ir, r in enumerate(r_range):
                if distance < r:
                    v_range[ir] += data.iat[i, j]
                    v_num[ir] += 1
                    break

    v_range_average = [v_range[i] / v_num[i] if v_num[i] != 0 else 0 for i in range(len(r_range))]
    return v_range_average


def generate_approximate_data(data, scale=1, center=(20, 20), grid_size=10, a=1.0):
    values = calculate_distances(data)

    new_shape = (data.shape[0] * scale, data.shape[1] * scale)
    approximated_data = pd.DataFrame(index=range(new_shape[0]), columns=range(new_shape[1]))
    scaled_center = (center[0] * scale, center[1] * scale)
    scaled_grid = grid_size / scale

    # print(f"values = {values}")

    # print(approximated_data.shape)

    # print(f"center*scale = {center*scale}, grid_size/scale = {grid_size/scale}")
    for i in range(approximated_data.shape[0]):
        for j in range(approximated_data.shape[1]):
            distance = calculate_distance_from_center(i, j, center=scaled_center, grid_size=scaled_grid, a=a)
            for ir, r in enumerate(r_range):
                if distance < r:
                    approximated_data.iat[i, j] = values[ir]
                    break
    return approximated_data.astype('float'), values


def calculate_error(original_data, approximated_data):
    total_diff = 0
    count = 0
    for i in range(original_data.shape[0]):
        for j in range(original_data.shape[1]):
            if not pd.isna(original_data.iat[i, j]):
                total_diff += abs(original_data.iat[i, j] - approximated_data.iat[i, j])
                count += 1
    return total_diff / count if count > 0 else float('inf')


def find_best_a(data):
    best_a = None
    best_error = float('inf')
    for a_val in np.arange(0.1, 2.1, 0.1):  # Iterating from 0.1 to 1.0
        approximated_data,_ = generate_approximate_data(data=data, scale=1, a=a_val)
        error = calculate_error(data, approximated_data)
        # print(f"error = {error}, best_error = {best_error}, a = {a_val}")
        if error < best_error:
            best_error = error
            best_a = a_val
    return best_a, best_error


file_list = glob.glob('data/All/XY_*.csv')
os.makedirs('data/fitted_results', exist_ok=True)

for file in file_list:
    if 'count' in file:
        continue
    data = load_csv(file)

    best_a, best_error = find_best_a(data)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plot_heat_map(data, axs[0], "Original Data")
    approximated_data, values = generate_approximate_data(data, a=best_a)
    plot_heat_map(approximated_data, axs[1], "Approximated Data")
    file_name = file.split('/')[-1].split('.')[0]
    plt.suptitle(f'{file} (a={best_a:.1f}), error={best_error:.2f}')

    file_save = file.replace('XY_', 'fitted_XY_').replace('csv', 'png').replace('/All/', '/fitted_results/')
    plt.savefig(file_save)
    plt.close()

    # save the values to a json
    result = dict()
    result['r_range'] = r_range
    result['values'] = values
    result['a'] = best_a
    result['b'] = b
    result['error'] = best_error
    json_file = file_save.replace('png', 'json')
    with open(json_file, 'w') as f:
        json.dump(result, f)
