import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import json
import os


def exp_decay_with_const(x, amplitude, decay_rate, offset):
    return amplitude * np.exp(-decay_rate * x) + offset


# root_path = 'data/Vehicle'
# root_path = 'data/ALL'
root_path = 'data/VRU'
output_dir = root_path + '/fitted_results'

file_list = glob.glob(root_path+'/time_*.csv')
os.makedirs(output_dir, exist_ok=True)

for file in file_list:
    if 'count' in file:
        continue
    file_name = file.split('/')[-1].split('.')[0]
    data = pd.read_csv(file, header=None)
    x_data = np.arange(0, len(data)*0.5, 0.5)
    y_data = data.values.flatten()

    params_exp, _ = curve_fit(exp_decay_with_const,
                              x_data, y_data, maxfev=10000)

    x_fit = np.linspace(0, len(y_data) - 1, 100)
    y_fit_exp = exp_decay_with_const(x_fit, *params_exp)

    plt.scatter(x_data, y_data, label='Data')
    plt.plot(x_fit, y_fit_exp, label='Exp. decay fit', color='red')
    plt.legend()
    plt.xlabel('time_interval(s)')
    plt.ylabel('value')
    plt.title('Curve Fitting for ' + file_name)
    formula_text = 'y = {:.2f} * exp(-{:.2f} * x) + {:.2f}'.format(*params_exp)
    plt.text(0.50, 0.50, formula_text,
             transform=plt.gca().transAxes, verticalalignment='top')

    file_save = file.replace('time_', 'fitted_time_').replace('csv', 'png').replace(root_path, output_dir)
    plt.savefig(file_save)
    plt.close()

    # save to json
    result = {'amplitude': params_exp[0], 'decay_rate': params_exp[1], 'offset': params_exp[2]}
    with open(file_save.replace('png', 'json'), 'w') as f:
        json.dump(result, f)
