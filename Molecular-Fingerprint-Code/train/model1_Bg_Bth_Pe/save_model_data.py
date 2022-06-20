import torch
import pandas as pd
import numpy as np

def save_model_and_data_chkpt(model, optimizer, epoch, train_y_cat, valid_y_cat, train_pred_cat, valid_pred_cat):
    
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
    }, "model_best_accuracy_chkpt.pt")
    
    train_y_data = train_y_cat.cpu().numpy()
    df_train_y = pd.DataFrame(train_y_data)
    df_train_y.to_csv("train_y_data_chkpt.csv",index=False)

    valid_y_data = valid_y_cat.cpu().numpy()
    df_valid_y = pd.DataFrame(valid_y_data)
    df_valid_y.to_csv("valid_y_data_chkpt.csv",index=False)

    train_pred_data = train_pred_cat.cpu().numpy()
    df_train_pred = pd.DataFrame(train_pred_data)
    df_train_pred.to_csv("train_pred_data_chkpt.csv",index=False)

    valid_pred_data = valid_pred_cat.cpu().numpy()
    df_valid_pred = pd.DataFrame(valid_pred_data)
    df_valid_pred.to_csv("valid_pred_data_chkpt.csv",index=False)

def save_model_and_data_end(model, optimizer, epoch, train_y_cat, valid_y_cat, train_pred_cat, valid_pred_cat):
    
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
    }, "model_end_train.pt")
   

    train_y_data = train_y_cat.cpu().numpy()
    df_train_y = pd.DataFrame(train_y_data)
    df_train_y.to_csv("train_y_data_end.csv",index=False)

    valid_y_data = valid_y_cat.cpu().numpy()
    df_valid_y = pd.DataFrame(valid_y_data)
    df_valid_y.to_csv("valid_y_data_end.csv",index=False)

    train_pred_data = train_pred_cat.cpu().numpy()
    df_train_pred = pd.DataFrame(train_pred_data)
    df_train_pred.to_csv("train_pred_data_end.csv",index=False)

    valid_pred_data = valid_pred_cat.cpu().numpy()
    df_valid_pred = pd.DataFrame(valid_pred_data)
    df_valid_pred.to_csv("valid_pred_data_end.csv",index=False)

def save_data_eval(eval_y_cat, eval_pred_cat):

    eval_y_data = eval_y_cat.cpu().numpy()
    df_eval_y = pd.DataFrame(eval_y_data)
    df_eval_y.to_csv("eval_y_data.csv",index=False)

    eval_pred_data = eval_pred_cat.cpu().numpy()
    df_eval_pred = pd.DataFrame(eval_pred_data)
    df_eval_pred.to_csv("eval_pred_data_end.csv",index=False)
