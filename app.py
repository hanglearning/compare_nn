import base64
from io import BytesIO

import math
import numpy as np
import torch

from flask import Flask
from matplotlib.figure import Figure
       
# import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

app = Flask(__name__)


@app.route("/")
def hello():
    nn1 = torch.load("/Users/chenhang/Downloads/m1")
    nn2 = torch.load("/Users/chenhang/Downloads/m2")
    layer_to_figs = compare_weights_change(nn1, nn2)

    layer_to_data = ""
    for layer, fig in layer_to_figs.items():
        buf = BytesIO()
        fig.savefig(buf, format="png")
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        layer_to_data += f"<img src='data:image/png;base64,{data}'/>\n"

    return layer_to_data

    # # Generate the figure **without using pyplot**.
    # fig = Figure()
    # ax = fig.subplots()
    # ax.plot([1, 2])
    # # Save it to a temporary buffer.
    # buf = BytesIO()
    # fig.savefig(buf, format="png")
    # # Embed the result in the html output.
    # data = base64.b64encode(buf.getbuffer()).decode("ascii")
    # return f"<img src='data:image/png;base64,{data}'/>"



def normalize_weights(nn, percent=0.2):

       layer_to_matrix = {}

       for name, param in nn.named_parameters():
            if 'weight' in name:
                param_1d_array = param.detach().numpy().flatten()

                # replace 0 with nan. 0 weights are most likely pruned
                param_1d_array[param_1d_array == 0] = np.nan

                percent_order = math.ceil(param_1d_array.size * percent)
                low_percent_threshold = np.partition(param_1d_array, percent_order)[percent_order]
                top_percent_threshold = np.partition(param_1d_array, -percent_order)[-percent_order]

                # reshape flatterned NN to 2D array with nan fillings
                side_len = math.ceil(math.sqrt(param_1d_array.size))

                param_2d_array = np.pad(param_1d_array.astype(float), (0, side_len*side_len - param_1d_array.size), 
                    mode='constant', constant_values=np.nan).reshape(side_len,side_len)

                # change top to 1
                param_2d_array[np.where(param_2d_array >= top_percent_threshold)] = 1

                # change middle elements to 0
                param_2d_array[np.where((param_2d_array > low_percent_threshold) & (param_2d_array < top_percent_threshold))] = 0

                # change low to -1
                param_2d_array[np.where(param_2d_array <= low_percent_threshold)] = -1

                layer_to_matrix[name] = param_2d_array
       return layer_to_matrix

def compare_weights_change(nn_1, nn_2, top_or_low = 1, percent=0.2):
    """
    Input: 
    nn1 is typically global_model or local_model_1
    nn2 is typically local_model or local_model_2
    top_or_low: 1 or -1; 1 for top percent, -1 for low percent
    """

    # fig, ax = plt.subplots(figsize=(18, 10))

    nn_1_layer_to_matrix = normalize_weights(nn_1, percent)
    nn_2_layer_to_matrix = normalize_weights(nn_2, percent)

    layer_to_figs = {}

    for layer, weights in nn_1_layer_to_matrix.items():

        fig = Figure()
        ax = fig.subplots()

        top_percent_same = np.logical_and(weights == top_or_low, nn_2_layer_to_matrix[layer] == top_or_low) + 0
        top_percent_old = np.logical_and(weights == top_or_low, nn_2_layer_to_matrix[layer] != top_or_low) + 0
        top_percent_new = np.logical_and(weights != top_or_low, nn_2_layer_to_matrix[layer] == top_or_low) + 0


        top_percent_new[top_percent_new == 1] = 3
        top_percent_same[top_percent_same == 1] = 2
        top_percent_old[top_percent_old == 1] = 1
        display_arr = top_percent_new + top_percent_same + top_percent_old

        # calculate change percentage
        entire_counts = (weights == top_or_low).sum()
        new_perc = (top_percent_new == 3).sum()/entire_counts
        same_perc = (top_percent_same == 2).sum()/entire_counts
        old_perc = (top_percent_old == 1).sum()/entire_counts
        
        # https://stackoverflow.com/questions/37719304/python-imshow-set-certain-value-to-defined-color
        # define color map 
        color_map = {0: np.array([255, 255, 255]), # while - nothing
                    1: np.array([0, 0, 255]), # blue - old
                    2: np.array([255, 0, 0]), # red - same
                    3: np.array([0, 255, 0])} # green - new 

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
        top_or_low_indicator = 'Top' if top_or_low == 1 else 'Low'
        patch_1 = mpatches.Patch(color='green', label=f'New - {new_perc:.2%}')
        patch_2 = mpatches.Patch(color='red', label=f'Same - {same_perc:.2%}')
        patch_3 = mpatches.Patch(color='blue', label=f'Old - {old_perc:.2%}')

        ax.legend(handles=[patch_1, patch_2, patch_3])

        fig.suptitle(f"{layer} - {top_or_low_indicator} {percent:.0%}", fontsize=20)
        # fig.savefig(f'/Users/chenhang/Downloads/weight_change/{layer}.png')
        layer_to_figs[layer] = fig

    return layer_to_figs
