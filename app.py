""" TODO -
1. auto form submit
2. remember last drop down manu
3. slider to slide rounds
4. make it original UI
5. select random model from file
"""


import base64
from io import BytesIO

import math
import numpy as np
import torch
import os
from os import listdir
from os.path import isfile, join
import pickle

from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap

from matplotlib.figure import Figure
       
import matplotlib.patches as mpatches

# import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

app = Flask(__name__)
Bootstrap(app)

models_base_folder = "/Users/chenhang/Downloads/models"

@app.route("/", methods=['GET', 'POST'])
def main():

    # populate selections
    rainable_layers, clients, selectable_rounds = get_selections(models_base_folder)

    # get selections from front-end
    client1 = request.form.get('clients1') if request.form.get('clients1') else "globals_0"
    client2 = request.form.get('clients2') if request.form.get('clients2') else "globals_0"
    round1 = request.form.get('rounds1') if request.form.get('rounds1') else 0
    round2 = request.form.get('rounds2') if request.form.get('rounds2') else 0
    top_or_low = request.form.get('top_or_low') if request.form.get('rounds2') else 'top'
    percent = float(request.form.get('percent')) if request.form.get('percent') else 0.2

    nn1_path = f"{models_base_folder}/{client1}/{round1}.pkl"
    nn2_path = f"{models_base_folder}/{client2}/{round2}.pkl"
    top_or_low_indicator = 3 if top_or_low == 'top' else 1
    
    
    left_figs = display_weights_single_net(nn1_path, 1, top_or_low_indicator, percent)
    middle_figs = compare_weights_change(nn1_path, nn2_path, top_or_low_indicator, percent)
    right_figs = display_weights_single_net(nn2_path, 2, top_or_low_indicator, percent)

    
    left_layer_to_plot_data = construct_layer_to_plot_data(left_figs)
    middle_layer_to_plot_data = construct_layer_to_plot_data(middle_figs)
    right_layer_to_plot_data = construct_layer_to_plot_data(right_figs)

    return render_template('index.html', 
                            rainable_layers = rainable_layers, clients = clients, selectable_rounds = selectable_rounds, 
                            client1 = client1, client2 = client2, round1 = round1, round2 = round2, top_or_low = top_or_low, percent = percent,
                            left_layer_to_plot_data = left_layer_to_plot_data, middle_layer_to_plot_data = middle_layer_to_plot_data, right_layer_to_plot_data = right_layer_to_plot_data)

def get_selections(models_base_folder):
    # get rainable layers 
    with open(f"{models_base_folder}/globals_0/0.pkl", 'rb') as f:
        ref_nn_layer_to_weights = pickle.load(f)
    
    # get clients
    clients = os.listdir(models_base_folder)
    clients.sort(key=lambda x: int(x.split('_')[1]))

    # get available comm rounds
    selectable_rounds = [f.split('.')[0] for f in listdir(f"{models_base_folder}/globals_0") if isfile(join(f"{models_base_folder}/globals_0", f))]
    selectable_rounds.sort(key=int)

    return list(ref_nn_layer_to_weights.keys()), clients, selectable_rounds

def construct_layer_to_plot_data(layer_to_figs):
    layer_to_data = []
    for layer, fig in layer_to_figs.items():
        buf = BytesIO()
        fig.savefig(buf, format="png")
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        # layer_to_data += f"<img src='data:image/png;base64,{data}'/><br><br>"
        layer_to_data.append(data)
    return layer_to_data

def normalize_weights(nn_path, percent=0.2):

    """ Transforms weights matrix to numbers:
        Top percent - 3
        Middles - 2
        Low percent - 1
        Pruned - (keep) 0

    Returns:
        layer_to_matrix _dict_: key: layer, value: transformed matrix
    """

    layer_to_matrix = {}

    with open(nn_path, 'rb') as f:
        nn_layer_to_weights = pickle.load(f)

    for layer, param in nn_layer_to_weights.items():
        
        param_1d_array = param.flatten()

        # replace 0 with nan. 0 weights are most likely pruned~~~ may not be necessary, also misleading when pruning
        # param_1d_array[param_1d_array == 0] = np.nan

        # take abs as we show magnitude values
        param_1d_array = np.absolute(param_1d_array)

        # create ref array to get threshold after removing 0 weights (which are pruned)
        non_0_param_1d_array = param_1d_array[param_1d_array != 0]

        percent_order = math.ceil(non_0_param_1d_array.size * percent)
        low_percent_threshold = np.partition(non_0_param_1d_array, percent_order)[percent_order]
        top_percent_threshold = np.partition(non_0_param_1d_array, -percent_order)[-percent_order]

        # reshape flatterned NN to 2D array and pad with -10 (originally as nan, but introduced difficulty in color mapping)
        side_len = math.ceil(math.sqrt(param_1d_array.size))

        param_2d_array = np.pad(param_1d_array.astype(float), (0, side_len*side_len - param_1d_array.size), 
            mode='constant', constant_values=-10).reshape(side_len,side_len)

        # change top weights to 3
        param_2d_array[np.where(param_2d_array >= top_percent_threshold)] = 3

        # change middle elements to 2
        param_2d_array[np.where((param_2d_array > low_percent_threshold) & (param_2d_array < top_percent_threshold))] = 2

        # change low weights to 1
        param_2d_array[np.where((0 < param_2d_array) & (param_2d_array <= low_percent_threshold))] = 1

        # keep pruned weights 0
        param_2d_array[param_2d_array == 0] = 0

        layer_to_matrix[layer.split(".")[0]] = param_2d_array

    return layer_to_matrix

def save_fig_to_disk():
    pass

def display_weights_single_net(nn_path, nn1_or_nn2, top_or_low = 3, percent=0.2):

    """ 
    nn1_or_nn2 -> int: 1 or 2
    top_or_low -> 1 or 3
    """

    nn_layer_to_matrix = normalize_weights(nn_path, percent)

    layer_to_figs = {}

    # nn1 -> green, nn2 -> blue
    weight_color_rgb = np.array([0, 0, 255]) if nn1_or_nn2 == 1 else np.array([0, 255, 0])
    weight_color = 'blue' if nn1_or_nn2 == 1 else 'green'

    for layer, weights in nn_layer_to_matrix.items():

        fig = Figure()
        ax = fig.subplots()

        color_map = {0: np.array([0, 0, 0]), # black - pruned
                     2: np.array([255, 255, 255]), # white - values in between,
                     -10: np.array([255, 255, 255]), # white - padding,
                    top_or_low: weight_color_rgb}
        
        if top_or_low == 1:
            color_map[3] = np.array([255, 255, 255])
        else:
            color_map[1] = np.array([255, 255, 255])

        # make a 3d numpy array that has a color channel dimension   
        data_3d = np.ndarray(shape=(weights.shape[0], weights.shape[1], 3), dtype=int)
        for i in range(0, weights.shape[0]):
            for j in range(0, weights.shape[1]):
                data_3d[i][j] = color_map[weights[i][j]]

        c = ax.matshow(data_3d,
                    interpolation ='nearest',
                    aspect='auto',
                        origin ='lower')

        client_index = nn_path.split("/")[-2]
        
        # fig.colorbar(c, ax = ax)
        top_or_low_indicator = 'Top' if top_or_low == 3 else 'Low'
        patch_1 = mpatches.Patch(color='black', label=f'Pruned')
        patch_2 = mpatches.Patch(color=weight_color, label=f'{top_or_low_indicator} {percent:.0%}')

        ax.legend(handles=[patch_1, patch_2])

        fig.suptitle(f"{layer} - {client_index} {top_or_low_indicator} {percent:.0%}", fontsize=20)

        # fig.savefig(f'/Users/chenhang/Downloads/{name}.png')
        
        layer_to_figs[layer] = fig

    return layer_to_figs

def compare_weights_change(nn_1_path, nn_2_path, top_or_low = 3, percent=0.2):
    """
    Input: 
    nn_1_path is typically the pkl path of global_model or local_model_1
    nn_2_path is typically the pkl path of local_model or local_model_2
    top_or_low: 3 or 1; 3 for top percent, 1 for low percent
    """

    # fig, ax = plt.subplots(figsize=(18, 10))

    nn_1_layer_to_matrix = normalize_weights(nn_1_path, percent)
    nn_2_layer_to_matrix = normalize_weights(nn_2_path, percent)

    layer_to_figs = {}

    for layer, weights in nn_1_layer_to_matrix.items():

        fig = Figure()
        ax = fig.subplots()

        same_weights = np.logical_and(weights == top_or_low, nn_2_layer_to_matrix[layer] == top_or_low) + 0
        nn1_unique = np.logical_and(weights == top_or_low, nn_2_layer_to_matrix[layer] != top_or_low) + 0
        nn2_unique = np.logical_and(weights != top_or_low, nn_2_layer_to_matrix[layer] == top_or_low) + 0


        nn2_unique[nn2_unique == 1] = 3
        same_weights[same_weights == 1] = 2
        nn1_unique[nn1_unique == 1] = 1
        display_arr = nn2_unique + same_weights + nn1_unique

        # calculate change percentage
        entire_counts = (weights == top_or_low).sum()
        nn2_perc = (nn2_unique == 3).sum()/entire_counts
        same_perc = (same_weights == 2).sum()/entire_counts
        nn1_perc = (nn1_unique == 1).sum()/entire_counts
        
        # https://stackoverflow.com/questions/37719304/python-imshow-set-certain-value-to-defined-color
        # define color map 
        color_map = {0: np.array([255, 255, 255]), # while - nothing
                    1: np.array([0, 0, 255]), # blue - nn1_unique
                    2: np.array([255, 0, 0]), # red - same
                    3: np.array([0, 255, 0])} # green - nn2_unique 

        # make a 3d numpy array that has a color channel dimension   
        data_3d = np.ndarray(shape=(display_arr.shape[0], display_arr.shape[1], 3), dtype=int)
        for i in range(0, display_arr.shape[0]):
            for j in range(0, display_arr.shape[1]):
                data_3d[i][j] = color_map[display_arr[i][j]]


        # print(display_arr)
        c = ax.matshow(data_3d,
                    interpolation ='nearest',
                    aspect='auto',
                        origin ='lower')

        
        # fig.colorbar(c, ax = ax)
        top_or_low_indicator = 'Top' if top_or_low == 3 else 'Low'
        patch_1 = mpatches.Patch(color='green', label=f'NN2 - {nn2_perc:.2%}')
        patch_2 = mpatches.Patch(color='red', label=f'Same - {same_perc:.2%}')
        patch_3 = mpatches.Patch(color='blue', label=f'NN1 - {nn1_perc:.2%}')

        ax.legend(handles=[patch_1, patch_2, patch_3])

        fig.suptitle(f"{layer} - {top_or_low_indicator} {percent:.0%}", fontsize=20)
        # fig.savefig(f'/Users/chenhang/Downloads/weight_change/{layer}.png')
        layer_to_figs[layer] = fig

    return layer_to_figs

if __name__ == "__main__":
    # hello()
    app.run(debug=True)