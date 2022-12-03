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
import sys

from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap

from matplotlib.figure import Figure
       
import matplotlib.patches as mpatches

# import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

app = Flask(__name__)
Bootstrap(app)

# models_base_folder = "/Users/chenhang/Downloads/models"
models_base_folder = "/Users/chenhang/Documents/Temp/11262022_011637_60/models"

@app.route("/", methods=['GET', 'POST'])
def main():

    # populate selections
    clients, selectable_rounds = get_selections(models_base_folder)

    # get selections from front-end
    clientA = request.form.get('clientsA') if request.form.get('clientsA') else "globals_0"
    clientB = request.form.get('clientsB') if request.form.get('clientsB') else "L_[3, 5, 7]_7"
    roundA = request.form.get('roundsA') if request.form.get('roundsA') else "R0"
    roundB = request.form.get('roundsB') if request.form.get('roundsB') else "R1_E10"
    top_or_low = request.form.get('top_or_low') if request.form.get('roundsB') else 'top'
    percent = float(request.form.get('percent')) if request.form.get('percent') else 0.2

    nn1_path = f"{models_base_folder}/{clientA}/{roundA}.pkl"
    nn2_path = f"{models_base_folder}/{clientB}/{roundB}.pkl"
    
    
    left_figs = display_weights_single_net(nn1_path, 1, top_or_low, percent)
    middle_figs = compare_weights_change(nn1_path, nn2_path, top_or_low, percent)
    right_figs = display_weights_single_net(nn2_path, 2, top_or_low, percent)

    
    left_layer_to_plot_data = construct_layer_to_plot_data(left_figs)
    middle_layer_to_plot_data = construct_layer_to_plot_data(middle_figs)
    right_layer_to_plot_data = construct_layer_to_plot_data(right_figs)

    return render_template('index.html', 
                            clients = clients, selectable_rounds = selectable_rounds, 
                            clientA = clientA, clientB = clientB, roundA = roundA, roundB = roundB, top_or_low = top_or_low, percent = percent,
                            left_layer_to_plot_data = left_layer_to_plot_data, middle_layer_to_plot_data = middle_layer_to_plot_data, right_layer_to_plot_data = right_layer_to_plot_data)

def get_selections(models_base_folder):
    
    # get clients
    clients = [name for name in os.listdir(models_base_folder) if os.path.isdir(os.path.join(models_base_folder, name))]
    clients.sort(key=lambda x: int(x.split('_')[-1]))

    # get available comm rounds for local models
    selectable_rounds = [f.split('.')[0] for f in listdir(f"{models_base_folder}/{clients[1]}") if isfile(join(f"{models_base_folder}/{clients[1]}", f))]
    selectable_rounds.sort(key=lambda x: (int(x.split('_')[0][1:]), int(x.split('_')[1].split(".")[0][1:])))

    # get available comm rounds for global models
    global_rounds = [f.split('.')[0] for f in listdir(f"{models_base_folder}/globals_0") if isfile(join(f"{models_base_folder}/globals_0", f))]
    global_rounds.sort(key=lambda x: int(x[1:]))
    selectable_rounds = global_rounds + selectable_rounds
   
    return clients, selectable_rounds

def construct_layer_to_plot_data(layer_to_figs):
    layer_to_data = []
    for layer, fig in layer_to_figs.items():
        buf = BytesIO()
        fig.savefig(buf, format="png")
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        # layer_to_data += f"<img src='data:image/png;base64,{data}'/><br><br>"
        layer_to_data.append(data)
    return layer_to_data


def normalize_weights(nn_path, top_or_low, percent=0.2):

    """ Transforms weights matrix to numbers:
        Top/Low weights - 2
        Others - 1
        Pruned - (keep) 0
        Padding - -10

    Returns:
        layer_to_matrix _dict_: key: layer, value: transformed matrix
    """

    layer_to_matrix = {}

    with open(nn_path, 'rb') as f:
        nn_layer_to_weights = pickle.load(f)

    for layer, param in nn_layer_to_weights.items():
        
        param_1d_array = param.flatten()

        # take abs as we show magnitude values
        param_1d_array = np.absolute(param_1d_array)

        pruned_percent = round(param_1d_array[param_1d_array == 0].size/param_1d_array.size, 1)
        
        # if need to display low, this is a trick
        display_percent = percent + pruned_percent if top_or_low == 'low' else percent
            
        percent_order = math.ceil(param_1d_array.size * display_percent)

        if top_or_low == 'top':
            percent_threshold = -np.sort(-param_1d_array)[percent_order]
        elif top_or_low == 'low':
            percent_threshold = np.sort(param_1d_array)[percent_order]

        # reshape flatterned NN to 2D array and pad with nan
        side_len = math.ceil(math.sqrt(param_1d_array.size))

        param_2d_array = np.pad(param_1d_array.astype(float), (0, side_len*side_len - param_1d_array.size), 
            mode='constant', constant_values=np.nan).reshape(side_len,side_len)

        display_2d_array = np.empty_like(param_2d_array)
        display_2d_array[:] = -10 # start from all paddings

        # keep pruned weights 0
        display_2d_array[param_2d_array == 0] = 0

        if top_or_low == 'top':
            # change top weights to 2
            display_2d_array[np.where(param_2d_array > percent_threshold)] = 2
            # change other weights to 1
            display_2d_array[np.where(param_2d_array < percent_threshold)] = 1
            # keep pruned weights 0, may overwrite some top weights if (1 - specified top show percent) > pruned_percent
            display_2d_array[param_2d_array == 0] = 0
        elif top_or_low == 'low':
            # change low weights to 2
            display_2d_array[np.where(param_2d_array < percent_threshold)] = 2
            # change other weights to 1
            display_2d_array[np.where(param_2d_array > percent_threshold)] = 1
            # keep pruned weights 0, may overwrite some low weights
            display_2d_array[param_2d_array == 0] = 0

        layer_to_matrix[layer.split(".")[0]] = display_2d_array.astype(int)

    return layer_to_matrix

def display_weights_single_net(nn_path, nn1_or_nn2, top_or_low = "top", percent=0.2):

    """ 
    nn1_or_nn2 -> int: 1 or 2
    """

    nn_layer_to_matrix = normalize_weights(nn_path, top_or_low, percent)

    layer_to_figs = {}

    # nn1 -> lime, nn2 -> blue
    weight_color_rgb = np.array([0, 0, 255]) if nn1_or_nn2 == 1 else np.array([0, 255, 0])
    weight_color = 'blue' if nn1_or_nn2 == 1 else 'lime'

    for layer, weights in nn_layer_to_matrix.items():

        fig = Figure()
        ax = fig.subplots()

        color_map = {0: np.array([0, 0, 0]), # black - pruned
                     1: np.array([247, 228, 194]), # skin color - other weights,
                     2: weight_color_rgb, # top/low weights, highlighted,
                     -10: np.array([255, 255, 255]) # white - padding,
                   }

        # make a 3d numpy array that has a color channel dimension   
        data_3d = np.ndarray(shape=(weights.shape[0], weights.shape[1], 3), dtype=int)
        for i in range(0, weights.shape[0]):
            for j in range(0, weights.shape[1]):
                try:
                    data_3d[i][j] = color_map[weights[i][j]]
                except:
                    print(weights)
                    sys.exit("Error in coloring.")
                    

        c = ax.matshow(data_3d,
                    interpolation ='nearest',
                    aspect='auto',
                        origin ='lower')

        # calculate percentage of highlights
        size_without_padding = weights[weights != -10].size
        pruned_percent = weights[weights == 0].size/size_without_padding
        other_percent = weights[weights == 1].size/size_without_padding
        highlighted_percent = weights[weights == 2].size/size_without_padding
        
        # fig.colorbar(c, ax = ax)
        patch_padding = mpatches.Patch(color='white', label=f'Padding')
        patch_0 = mpatches.Patch(color='black', label=f'Pruned {pruned_percent:.0%}')
        patch_1 = mpatches.Patch((247/256, 228/256, 194/256), label=f'Others {other_percent:.0%}')
        patch_2 = mpatches.Patch(color=weight_color, label=f'{top_or_low.title()} {highlighted_percent:.0%}')

        ax.legend(handles=[patch_padding, patch_0, patch_1, patch_2])

        client_index = nn_path.split("/")[-2]
        fig.suptitle(f"{layer} - {client_index} {top_or_low.title()} {percent:.0%}", fontsize=20)

        # fig.savefig(f"{nn_path.split('.')[0]}.png")
        
        layer_to_figs[layer] = fig

    return layer_to_figs

def compare_weights_change(nn_1_path, nn_2_path, top_or_low, percent=0.2):
    """
    Input: 
    nn_1_path is typically the pkl path of global_model or local_model_1
    nn_2_path is typically the pkl path of local_model or local_model_2
    Output:
        plot display_arr: -1, 0, 1, 2, 3 - see color map
                        
    """

    # fig, ax = plt.subplots(figsize=(18, 10))

    nn_1_layer_to_matrix = normalize_weights(nn_1_path, top_or_low, percent)
    nn_2_layer_to_matrix = normalize_weights(nn_2_path, top_or_low, percent)

    layer_to_figs = {}

    for layer, weights in nn_1_layer_to_matrix.items():

        fig = Figure()
        ax = fig.subplots()

        same_weights = np.logical_and(weights == 2, nn_2_layer_to_matrix[layer] == 2) + 0
        nn1_unique = np.logical_and(weights == 2, nn_2_layer_to_matrix[layer] == 1) + 0
        nn2_unique = np.logical_and(weights == 1, nn_2_layer_to_matrix[layer] == 2) + 0


        nn2_unique[nn2_unique == 1] = 3
        same_weights[same_weights == 1] = 2
        nn1_unique[nn1_unique == 1] = 1
        display_arr = nn2_unique + same_weights + nn1_unique

        # show pruned weights as well
        display_arr[weights == 0] = -1
        display_arr[nn_2_layer_to_matrix[layer] == 0] = -1

        # calculate change percentage
        entire_counts = (weights == 2).sum() # same amount for both networks, due to same percent
        nn2_perc = (nn2_unique == 3).sum()/entire_counts
        same_perc = (same_weights == 2).sum()/entire_counts
        nn1_perc = (nn1_unique == 1).sum()/entire_counts
        
        # https://stackoverflow.com/questions/37719304/python-imshow-set-certain-value-to-defined-color
        # define color map 
        color_map = {-1: np.array([0, 0, 0]), #  black - pruned
                    0: np.array([255, 255, 255]), # white - nothing
                    1: np.array([0, 0, 255]), # blue - nn1_unique
                    2: np.array([255, 0, 0]), # red - same
                    3: np.array([0, 255, 0])} # lime - nn2_unique 

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
        patch_padding = mpatches.Patch(color='white', label=f'Padding')
        patch_0 = mpatches.Patch(color='black', label=f'Pruned')
        patch_1 = mpatches.Patch(color='lime', label=f'NN2 - {nn2_perc:.2%}')
        patch_2 = mpatches.Patch(color='red', label=f'Same - {same_perc:.2%}')
        patch_3 = mpatches.Patch(color='blue', label=f'NN1 - {nn1_perc:.2%}')

        ax.legend(handles=[patch_padding, patch_0, patch_1, patch_2, patch_3])

        fig.suptitle(f"{layer} - {top_or_low.title()} {percent:.0%}", fontsize=20)
        # fig.savefig(f'/Users/chenhang/Downloads/weight_change/{layer}.png')
        layer_to_figs[layer] = fig

    return layer_to_figs

if __name__ == "__main__":
    app.run(debug=True)