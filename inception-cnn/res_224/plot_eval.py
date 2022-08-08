import matplotlib.pyplot as plt
import torch
import numpy as np
import scaling_torch_lib as mike

def plot_data(eval_y, eval_pred):

    bg_true, bth_true, pe_true = mike.unnormalize_params_plot(eval_y)
    bg_pred, bth_pred, pe_pred = mike.unnormalize_params_plot(eval_pred)

    bg_true = bg_true.cpu().numpy()
    bg_pred = bg_pred.cpu().numpy()

    bth_true = bth_true.cpu().numpy()
    bth_pred = bth_pred.cpu().numpy()

    pe_true = pe_true.cpu().numpy()
    pe_pred = pe_pred.cpu().numpy()

    max_bg = np.amax(np.maximum(bg_true,bg_pred)) 
    max_bth = np.amax(np.maximum(bth_true,bth_pred)) 
    max_pe = np.amax(np.maximum(pe_true,pe_pred)) 

    max_bg_figure = max_bg * 1.1
    max_bth_figure = max_bth * 1.1
    max_pe_figure = max_pe * 1.1

    bg_line = (0, max_bg)
    bth_line = (0, max_bth)
    pe_line = (0, max_pe)

    athermal_mask = bg_true < bth_true**0.824

    bth_true_ath = bth_true[athermal_mask]
    bth_pred_ath = bth_pred[athermal_mask]

    bth_true_g = bth_true[~athermal_mask]
    bth_pred_g = bth_pred[~athermal_mask]

    bg_true_ath = bg_true[athermal_mask]
    bg_pred_ath = bg_pred[athermal_mask]

    bg_true_g = bg_true[~athermal_mask]
    bg_pred_g = bg_pred[~athermal_mask]

    pe_true_ath = pe_true[athermal_mask]
    pe_pred_ath = pe_pred[athermal_mask]

    pe_true_g = pe_true[~athermal_mask]
    pe_pred_g = pe_pred[~athermal_mask]

    fig, (ax_bg, ax_bth, ax_pe) = plt.subplots(1,3,figsize=(12,3))
    ax_bg.scatter(bg_true_ath, bg_pred_ath, c='r', s=8)
    ax_bg.scatter(bg_true_g, bg_pred_g, c='b', s=8)
    ax_bth.scatter(bth_true_ath, bth_pred_ath,c='r', s=8)
    ax_bth.scatter(bth_true_g, bth_pred_g,c='b', s=8)
    ax_pe.scatter(pe_true_ath, pe_pred_ath,c='r', s=8)
    ax_pe.scatter(pe_true_g, pe_pred_g,c='b', s=8)

    ax_bg.plot([0,max_bg_figure], [0,max_bg_figure], 'k-')
    ax_bth.plot([0,max_bth_figure], [0,max_bth_figure], 'k-')
    ax_pe.plot([0,max_pe_figure], [0,max_pe_figure], 'k-')

    plots = [ax_bg, ax_bth, ax_pe]
    for plot in plots:
        plot.set_xlabel('true value')
        plot.set_ylabel('predicted value')
        plot.set_aspect('equal')

    ax_bg.set_xlim(0, max_bg_figure)
    ax_bg.set_ylim(0, max_bg_figure)
    ax_bg.set_title('Bg')

    ax_bth.set_xlim(0, max_bth_figure)
    ax_bth.set_ylim(0, max_bth_figure)
    ax_bth.set_title('Bth')

    ax_pe.set_xlim(0, max_pe_figure)
    ax_pe.set_ylim(0, max_pe_figure)
    ax_pe.set_title('Pe')

    plt.show()

    plt.close()
