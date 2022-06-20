import matplotlib.pyplot as plt
import torch
import numpy as np
import scaling_torch_lib as mike

def plot_data(eval_y, eval_pred):

    bg_true, pe_true = mike.unnormalize_params(eval_y)
    bg_pred, pe_pred = mike.unnormalize_params(eval_pred)

    bg_true = bg_true.cpu().numpy()
    bg_pred = bg_pred.cpu().numpy()

    pe_true = pe_true.cpu().numpy()
    pe_pred = pe_pred.cpu().numpy()

    max_bg = np.amax(np.maximum(bg_true,bg_pred)) 
    max_pe = np.amax(np.maximum(pe_true,pe_pred)) 

    max_bg_figure = max_bg * 1.1
    max_pe_figure = max_pe * 1.1

    bg_line = (0, max_bg)
    pe_line = (0, max_pe)

    fig, (ax_bg, ax_pe) = plt.subplots(1,2,figsize=(8,2))
    ax_bg.scatter(bg_true, bg_pred, c='b', s=8)
    ax_pe.scatter(pe_true, pe_pred,c='b', s=8)

    ax_bg.plot([0,max_bg_figure], [0,max_bg_figure], 'k-')
    ax_pe.plot([0,max_pe_figure], [0,max_pe_figure], 'k-')

    plots = [ax_bg, ax_pe]
    for plot in plots:
        plot.set_xlabel('true value')
        plot.set_ylabel('predicted value')
        plot.set_aspect('equal')

    ax_bg.set_xlim(0, max_bg_figure)
    ax_bg.set_ylim(0, max_bg_figure)
    ax_bg.set_title('Bg')

    ax_pe.set_xlim(0, max_pe_figure)
    ax_pe.set_ylim(0, max_pe_figure)
    ax_pe.set_title('Pe')

    plt.show()

    plt.close()
