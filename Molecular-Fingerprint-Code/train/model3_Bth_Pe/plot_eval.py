import matplotlib.pyplot as plt
import torch
import numpy as np
import scaling_torch_lib as mike

def plot_data(eval_y, eval_pred):

    bth_true, pe_true = mike.unnormalize_params(eval_y)
    bth_pred, pe_pred = mike.unnormalize_params(eval_pred)

    bth_true = bth_true.cpu().numpy()
    bth_pred = bth_pred.cpu().numpy()

    pe_true = pe_true.cpu().numpy()
    pe_pred = pe_pred.cpu().numpy()

    max_bth = np.amax(np.maximum(bth_true,bth_pred)) 
    max_pe = np.amax(np.maximum(pe_true,pe_pred)) 

    max_bth_figure = max_bth * 1.1
    max_pe_figure = max_pe * 1.1

    bth_line = (0, max_bth)
    pe_line = (0, max_pe)

    fig, (ax_bth, ax_pe) = plt.subplots(1,2,figsize=(8,2))
    ax_bg.scatter(bth_true, bth_pred, c='b', s=8)
    ax_pe.scatter(pe_true, pe_pred,c='b', s=8)

    ax_bth.plot([0,max_bth_figure], [0,max_bth_figure], 'k-')
    ax_pe.plot([0,max_pe_figure], [0,max_pe_figure], 'k-')

    plots = [ax_bth, ax_pe]
    for plot in plots:
        plot.set_xlabel('true value')
        plot.set_ylabel('predicted value')
        plot.set_aspect('equal')

    ax_bth.set_xlim(0, max_bth_figure)
    ax_bth.set_ylim(0, max_bth_figure)
    ax_bth.set_title('Bth')

    ax_pe.set_xlim(0, max_pe_figure)
    ax_pe.set_ylim(0, max_pe_figure)
    ax_pe.set_title('Pe')

    plt.show()

    plt.close()
