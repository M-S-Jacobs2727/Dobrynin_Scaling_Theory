import numpy as np
import torch
import scaling_torch_lib as mike
import time
import pandas as pd
import save_model_data as savechkpt
import torch.nn.functional as F
from torch.jit.annotations import Optional, Tuple
import test_network_serial as test_gen 
import collections

pd.options.mode.chained_assignment = None  #

torch.cuda.empty_cache()

dtype=torch.float

Param = collections.namedtuple('Param', ('min', 'max'))

PHI = Param(3e-5, 2e-2)
NW = Param(100, 1e5)
ETA_SP = Param(1, 1e6)

BG = Param(0.36, 1.55)
BTH = Param(0.22, 0.82)
PE = Param(2.5, 13.5)

ETA_SP_131 = Param(ETA_SP.min/NW.max/PHI.max**(1/(3*0.588-1)),ETA_SP.max/NW.min/PHI.min**(1/(3*0.588-1)))
ETA_SP_2 = Param(ETA_SP.min/NW.max/PHI.max**2,ETA_SP.max/NW.min/PHI.min**2)

NUM_BIN = 224

class Inception3(torch.nn.Module):

    def __init__(self, num_classes=1, aux_logits=False, transform_input=False, inception_blocks=None, init_weights=None):
        super(Inception3, self).__init__()
        if inception_blocks is None:
            inception_blocks = [BasicConv2d, InceptionA, InceptionB, InceptionC, InceptionD, InceptionE, InceptionAux]
        assert len(inception_blocks) == 7
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block(1, 32, kernel_size=3, stride=2)                 # Input of 1 channel instead of 3 channels
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        self.Mixed_6e = inception_c(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes)
        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(2048, num_classes)

    def _transform_input(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x):

        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        aux_defined = self.training and self.aux_logits
        if aux_defined:
            aux = self.AuxLogits(x)
        else:
            aux = None

        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, aux

    @torch.jit.unused
    def eager_outputs(self, x, aux):
        if self.training and self.aux_logits:
            return InceptionOutputs(x, aux)
        else:
            return x

    def forward(self, x):
        x = self._transform_input(x)
        x, aux = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.ward("Scripted Inception3 always returns Inception3 Tuple")
            return InceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)

class InceptionA(torch.nn.Module):

    def __init__(self, in_channels, pool_features, conv_block=None):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size = 1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size = 3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size = 3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionB(torch.nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size = 1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size = 3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size = 3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionC(torch.nn.Module):

    def __init__(self, in_channels, channels_7x7, conv_block=None):
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1,7), padding=(0,3))
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7,1), padding=(3,0))

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size = 1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size =(7,1), padding=(3,0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size =(1,7), padding=(0,3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size =(7,1), padding=(3,0))
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size =(1,7), padding=(0,3))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionD(torch.nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1,7), padding=(0,3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7,1), padding=(3,0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionE(torch.nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1,3), padding=(0,1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3,1), padding=(1,0))

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size = 1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size =(1,3), padding=(0,1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size =(3,1), padding=(1,0))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionAux(torch.nn.Module):

    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = torch.nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        x = self.conv0(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class BasicConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x, inplace=True)

def train( 
        model, loss_fn, optimizer, device,
        num_samples, batch_size, resolution):
    model.train()
    num_batches = num_samples // batch_size
    avg_loss = 0

    train_pred = torch.zeros(num_samples, 1).to(device)
    train_y = torch.zeros(num_samples, 1).to(device)

    train_Nw_min = np.zeros(num_samples)
    train_Nw_max = np.zeros(num_samples)
    train_num_Nw = np.zeros(num_samples)

    counter = 0

    for b, (X, y, Nw_min, Nw_max, Num_Nw) in enumerate(mike.surface_generator_Bth(num_batches, batch_size, device, return_nw=True, resolution=resolution)):

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        avg_loss += loss

        train_pred[counter:counter+batch_size] = pred
        train_y[counter:counter+batch_size] = y

        train_Nw_min[counter:counter+batch_size] = Nw_min
        train_Nw_max[counter:counter+batch_size] = Nw_max
        train_num_Nw[counter:counter+batch_size] = Num_Nw

        counter = counter + batch_size

    avg_loss/=num_batches
    
    return avg_loss, train_pred, train_y, train_Nw_min, train_Nw_max, train_num_Nw

def validate( 
        model, loss_fn, device,
        num_samples, batch_size, resolution):
    model.eval()
    avg_loss = 0

    all_data_pred = torch.zeros(num_samples, 1).to(device)
    all_data_y = torch.zeros(num_samples, 1).to(device)

    counter = 0

    num_batches = num_samples // batch_size

    valid_Nw_min = np.zeros(num_samples)
    valid_Nw_max = np.zeros(num_samples)
    valid_num_Nw = np.zeros(num_samples)

    with torch.no_grad():
        for b, (X, y, Nw_min, Nw_max, Num_Nw) in enumerate(mike.surface_generator_Bth(num_batches, batch_size, device, return_nw=True, resolution=resolution)):
            pred = model(X)
            loss = loss_fn(pred, y)
            avg_loss += loss

            all_data_pred[counter:counter+batch_size] = pred
            all_data_y[counter:counter+batch_size] = y
            valid_Nw_min[counter:counter+batch_size] = Nw_min
            valid_Nw_max[counter:counter+batch_size] = Nw_max
            valid_num_Nw[counter:counter+batch_size] = Num_Nw

            counter = counter + batch_size
    
    avg_loss /= num_batches

    return avg_loss, all_data_pred, all_data_y, valid_Nw_min, valid_Nw_max, valid_num_Nw

def test_model(model, device, resolution, X_2, table_vals):

    model.eval()
    with torch.no_grad():
        pred = model(X_2)

    bth_pred = mike.unnormalize_Bth_param(pred)

    table_vals["Pred Bth"] = bth_pred.tolist()

    bth_error = test_gen.get_error_bth(table_vals)

    return bth_error

def main():

    batch_size = 64
    train_size = 51200
    test_size = 21952
    eval_size = test_size
    epochs = 300
    epoch_best_accuracy = 0
    resolution = (224, 224)

    path_read = "/proj/avdlab/projects/Solutions_ML/exp-data/"
    df = pd.read_csv(f"{path_read}exp-data-rcs.csv")

    epoch_best_accuracy = 0
    epoch_best_loss = 0
    epoch_best_exp = 100

    model = Inception3()
 
    device = torch.device(f'cuda:2')
    model.to(device)
    print(f'{device = }')

    # init data table of all systems
    table_vals = test_gen.get_all_systems_table(df, device)
    X_131, X_2 = test_gen.load_exp_data(df, device, resolution)
    X_131 = X_131.to(device)
    X_2 = X_2.to(device)

    loss_fn = torch.nn.MSELoss()
    print(f'Epoch\ttrain_loss\tBth Train Error\ttest_loss\tBth Test Error\tBth Exp Data Error\ttime', flush=True)

    best_ave_accuracy = 100
    best_validation_loss = 100
    optimizer = torch.optim.Adam(model.parameters(),
    lr=0.00015226109872869536,
    betas=(0.7269928273673115, 0.8270538401297101),
    weight_decay=0,
    eps=1.7047882868450976e-08
    )

    init_train = np.zeros(train_size)
    init_valid = np.zeros(test_size)

    df_train_accuracy = pd.DataFrame(
            {
                "Bth-Train-True": pd.Series(data=init_train,dtype="float"),
                "Bth-Train-Pred": pd.Series(data=init_train,dtype="float"),
                "Nw-min": pd.Series(data=init_train,dtype="float"),
                "Nw-max": pd.Series(data=init_train,dtype="float"),
                "Num-Nw": pd.Series(data=init_train,dtype="int"),
             }
        )

    df_valid_accuracy = pd.DataFrame(
            {
                "Bth-Valid-True": pd.Series(data=init_valid,dtype="float"),
                "Bth-Valid-Pred": pd.Series(data=init_valid,dtype="float"),
                "Nw-min": pd.Series(data=init_valid,dtype="float"),
                "Nw-max": pd.Series(data=init_valid,dtype="float"),
                "Num-Nw": pd.Series(data=init_valid,dtype="int"),
             }
        )

    df_train_loss = pd.DataFrame(
            {
                "Bth-Train-True": pd.Series(data=init_train,dtype="float"),
                "Bth-Train-Pred": pd.Series(data=init_train,dtype="float"),
                "Nw-min": pd.Series(data=init_train,dtype="float"),
                "Nw-max": pd.Series(data=init_train,dtype="float"),
                "Num-Nw": pd.Series(data=init_train,dtype="int"),
             }
        )

    df_valid_loss = pd.DataFrame(
            {
                "Bth-Valid-True": pd.Series(data=init_valid,dtype="float"),
                "Bth-Valid-Pred": pd.Series(data=init_valid,dtype="float"),
                "Nw-min": pd.Series(data=init_valid,dtype="float"),
                "Nw-max": pd.Series(data=init_valid,dtype="float"),
                "Num-Nw": pd.Series(data=init_valid,dtype="int"),
             }
        )

    for m in range(epochs):

        t_start = time.perf_counter()

        train_loss, train_pred, train_y, train_Nw_min, train_Nw_max, train_num_Nw = train( 
            model, loss_fn, optimizer, device,
            train_size, batch_size, resolution
        )

        bth_train_true = mike.unnormalize_Bth_param(train_y)
        bth_train_pred = mike.unnormalize_Bth_param(train_pred)

        test_loss, valid_pred, valid_y, valid_Nw_min, valid_Nw_max, valid_num_Nw = validate(
            model, loss_fn, device,
            test_size, batch_size, resolution
        )

        test_exp_data_error = test_model(model, device, resolution, X_2, table_vals)

        bth_valid_true = mike.unnormalize_Bth_param(valid_y)
        bth_valid_pred = mike.unnormalize_Bth_param(valid_pred)

        bth_train_error = torch.mean(torch.abs(bth_train_true-bth_train_pred)/bth_train_true)

        bth_valid_error = torch.mean(torch.abs(bth_valid_true-bth_valid_pred)/bth_valid_true)

        elapsed = time.perf_counter() - t_start

        print(f'{m}\t{train_loss:>8f}\t{bth_train_error:.4f}\t{test_loss:>8f}\t{bth_valid_error:.4f}\t{test_exp_data_error:.4f}\t{elapsed:.2f}', flush=True)

        if test_exp_data_error < epoch_best_exp:

            epoch_best_exp = test_exp_data_error
            epoch_best_exp_error = m+1
            torch.save({
            'epoch': epoch_best_exp_error,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, "model_best_exp_error.pt")

        if (bth_train_error+bth_valid_error)/2.0 <= best_ave_accuracy:

            best_ave_accuracy = (bth_train_error+bth_valid_error)/2.0
            epoch_best_accuracy = m+1
            torch.save({
            'epoch': epoch_best_accuracy,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, "model_best_accuracy.pt")

            df_train_accuracy["Bth-Train-True"] = bth_train_true.detach().cpu().numpy()
            df_train_accuracy["Bth-Train-Pred"] = bth_train_pred.detach().cpu().numpy()
            df_train_accuracy["Nw-min"] = train_Nw_min
            df_train_accuracy["Nw-max"] = train_Nw_max
            df_train_accuracy["Num-Nw"] = train_num_Nw

            df_valid_accuracy["Bth-Valid-True"] = bth_valid_true.detach().cpu().numpy()
            df_valid_accuracy["Bth-Valid-Pred"] = bth_valid_pred.detach().cpu().numpy()
            df_valid_accuracy["Nw-min"] = valid_Nw_min
            df_valid_accuracy["Nw-max"] = valid_Nw_max
            df_valid_accuracy["Num-Nw"] = valid_num_Nw

            df_train_accuracy.to_csv('df_train_accuracy.csv', index=False)
            df_valid_accuracy.to_csv('df_valid_accuracy.csv', index=False)

        if test_loss < best_validation_loss:
            epoch_best_loss = m+1
            best_validation_loss = test_loss
            torch.save({
            'epoch': epoch_best_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, "model_lowest_loss.pt")

            df_train_loss["Bth-Train-True"] = bth_train_true.detach().cpu().numpy()
            df_train_loss["Bth-Train-Pred"] = bth_train_pred.detach().cpu().numpy()
            df_train_loss["Nw-min"] = train_Nw_min
            df_train_loss["Nw-max"] = train_Nw_max
            df_train_loss["Num-Nw"] = train_num_Nw

            df_valid_loss["Bth-Valid-True"] = bth_valid_true.detach().cpu().numpy()
            df_valid_loss["Bth-Valid-Pred"] = bth_valid_pred.detach().cpu().numpy()
            df_valid_loss["Nw-min"] = valid_Nw_min
            df_valid_loss["Nw-max"] = valid_Nw_max
            df_valid_loss["Num-Nw"] = valid_num_Nw

            df_train_loss.to_csv('df_train_loss.csv', index=False)
            df_valid_loss.to_csv('df_valid_loss.csv', index=False)

    torch.save({
    'epoch': m+1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
    }, "model_end.pt")

    print(f'{epoch_best_accuracy = }', flush=True)
    print(f'{epoch_best_loss = }', flush=True)

if __name__ == '__main__':

    main()
