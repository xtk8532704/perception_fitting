import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

a = 0.4
b = 1.0

r_range = [10, 20, 40, 60, 80, 120, 150, 180, 1000]

def load_csv(file_path):
    data = pd.read_csv(file_path, header=None, skiprows=1, usecols=range(1, 42))
    
    # Check if data is 41x41 after skipping the first row and column
    if data.shape != (41, 41):
        raise ValueError("CSV file does not contain 40x40 data after skipping the first row and column")
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if pd.isna(data.iat[i, j]):
                data.iat[i, j] = 0.0

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


def calculate_distance_from_center(i, j, center=(20, 20), grid_size=10):
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


def generate_approximate_data(data, scale=3, center=(20, 20), grid_size=10):
    values = calculate_distances(data)

    new_shape = (data.shape[0] * scale, data.shape[1] * scale)
    approximated_data = pd.DataFrame(index=range(new_shape[0]), columns=range(new_shape[1]))
    scaled_center = (center[0] * scale, center[1] * scale)
    scaled_grid = grid_size / scale

    print(f"values = {values}")

    print(approximated_data.shape)

    print(f"center*scale = {center*scale}, grid_size/scale = {grid_size/scale}")
    for i in range(approximated_data.shape[0]):
        for j in range(approximated_data.shape[1]):
            distance = calculate_distance_from_center(i, j, center=scaled_center, grid_size=scaled_grid)
            for ir, r in enumerate(r_range):
                if distance < r:
                    approximated_data.iat[i, j] = values[ir]
                    break
    return approximated_data.astype('float')

file_path = '/home/horibe/workspace/perception-performance-approximation/data/tp_rate.csv'  # Replace with your file path
data = load_csv(file_path)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
plot_heat_map(data, axs[0], "Original Data")
approximated_data = generate_approximate_data(data)
plot_heat_map(approximated_data, axs[1], "Approximated Data")
plt.suptitle('TP Rate')
plt.show()