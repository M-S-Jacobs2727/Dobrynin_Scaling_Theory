import collections
import numpy as np
import pandas as pd
import torch
import random

Param = collections.namedtuple('Param', ('min', 'max'))

PHI = Param(3e-5, 2e-2)
NW = Param(100, 1e5)
ETA_SP = Param(1, 1e6)

BG = Param(0.36, 1.55)
BTH = Param(0.22, 0.82)
PE = Param(2.5, 13.5)

# theoretical limits based on ETA_SP:
ETA_SP_131 = Param(ETA_SP.min/NW.max/PHI.max**(1/(3*0.588-1)),ETA_SP.max/NW.min/PHI.min**(1/(3*0.588-1)))
ETA_SP_2 = Param(ETA_SP.min/NW.max/PHI.max**2,ETA_SP.max/NW.min/PHI.min**2)

def state_diagram(df_all, df_table, device):
    """ Output table that can be used to plot state diagram
    """
    print(df_all)

    df2 = df_all.copy()
    df2 = df2[(df2['Nw'] >= NW.min) & (df2['Nw'] <= NW.max)]
    df2 = df2[(df2['phi'] >= PHI.min) & (df2['phi'] <= PHI.max)]
    df2 = df2[(df2['eta_sp'] > ETA_SP.min) & (df2['eta_sp'] < ETA_SP.max)]
    df2 = df2[(df2['Bg'] > 0) | (df2['Bth'] > 0)]

    df_table2 = df_table.copy()
    num_points = np.sum(df_table2["Num Nw"])
    init = np.zeros(num_points)

    df_create = pd.DataFrame(
            {
                "Polymer": pd.Series(data=init,dtype="str"),
                "Solvent": pd.Series(data=init,dtype="str"),
                "Group": pd.Series(data=init,dtype="int"),
                "True Bg": pd.Series(data=init,dtype="float"),
                "Pred Bg": pd.Series(data=init,dtype="float"),
                "True Bth": pd.Series(data=init,dtype="float"),
                "Pred Bth": pd.Series(data=init,dtype="float"),
                "Nw": pd.Series(data=init,dtype="float"),
                "phi_star": pd.Series(data=init,dtype="float"),
                "phi_star_star": pd.Series(data=init,dtype="float"),
                "l": pd.Series(data=init,dtype="float"),
                "b": pd.Series(data=init,dtype="float"),
                "v": pd.Series(data=init,dtype="float"),
                "phi_x": pd.Series(data=init,dtype="float"),
                "len_y": pd.Series(data=init,dtype="float"),
             }
        )

    counter = 0

    for i in np.unique(df_table2['Group']):

        Bg = np.unique(df_table2[df_table2["Group"] == i]["Pred Bg"])[0]
        Bth = np.unique(df_table2[df_table2["Group"] == i]["Pred Bth"])[0]
        l = np.unique(df2[df2["group"] == i]["l"])[0]
        phi_th = Bth**3*(Bth/Bg)**(1/(2*0.588-1))
        phi_star_star = Bth**4
        b = l/Bth**2
        
        mask = df2.loc[:, "group"] == i
        Nw_vals = np.unique(df2.loc[mask, "Nw"])

        if len(Nw_vals) == 1:

            Nw = np.unique(df2[df2["group"] == i]["Nw"])[0]

            df_create["Polymer"][counter] = np.unique(df_table2[df_table2["Group"] == i]["Polymer"])[0]
            df_create["Solvent"][counter] = np.unique(df_table2[df_table2["Group"] == i]["Solvent"])[0]
            df_create["True Bg"][counter] = np.unique(df_table2[df_table2["Group"] == i]["True Bg"])[0]
            df_create["Pred Bg"][counter] = Bg 
            df_create["True Bth"][counter] = np.unique(df_table2[df_table2["Group"] == i]["True Bth"])[0]
            df_create["Pred Bth"][counter] = Bth
            df_create["Nw"][counter] = Nw
            df_create["phi_star_star"][counter] = phi_star_star
            df_create["l"][counter] = l
            df_create["b"][counter] = b

            if df_create["True Bg"][counter] > 0:
                phi_star = Bg**3*Nw**(1-3*0.588)
            else:
                phi_star = Bth**3*Nw**(-0.5)

            df_create["phi_star"][counter] = phi_star

            v = b*l**2*phi_th/phi_star_star
            df_create["v"][counter] = v

            df_create["phi_x"][counter] = phi_star/phi_star_star
            df_create["len_y"][counter] = v/b/l**2

            counter += 1

        else:

            for Nw in Nw_vals:

                df_create["Polymer"][counter] = np.unique(df_table2[df_table2["Group"] == i]["Polymer"])[0]
                df_create["Solvent"][counter] = np.unique(df_table2[df_table2["Group"] == i]["Solvent"])[0]
                df_create["True Bg"][counter] = np.unique(df_table2[df_table2["Group"] == i]["True Bg"])[0]
                df_create["Pred Bg"][counter] = Bg
                df_create["True Bth"][counter] = np.unique(df_table2[df_table2["Group"] == i]["True Bth"])[0]
                df_create["Pred Bth"][counter] = Bth
                df_create["Nw"][counter] = Nw
                df_create["phi_star_star"][counter] = phi_star_star
                df_create["l"][counter] = l
                df_create["b"][counter] = b

                if df_create["True Bg"][counter] > 0:
                    phi_star = Bg**3*Nw**(1-3*0.588)
                else:
                    phi_star = Bth**3*Nw**(-0.5)
                df_create["phi_star"][counter] = phi_star

                v = b*l**2*phi_th/phi_star_star
                df_create["v"][counter] = v

                df_create["phi_x"][counter] = phi_star/phi_star_star
                df_create["len_y"][counter] = v/b/l**2

                counter += 1

    df_create.to_csv("df_state_diagram.csv")
    print(df_create)

def universal_plot(df_all, df_table, device):
    """ Output table that can be used to plot universal plot
    """

    df2 = df_all.copy()
    df2 = df2[(df2['Nw'] >= NW.min) & (df2['Nw'] <= NW.max)]
    df2 = df2[(df2['phi'] >= PHI.min) & (df2['phi'] <= PHI.max)]
    df2 = df2[(df2['eta_sp'] > ETA_SP.min) & (df2['eta_sp'] < ETA_SP.max)]
    df2 = df2[(df2['Bg'] > 0) | (df2['Bth'] > 0)]

    df_table2 = df_table.copy()
    #num_points = len(df2)
    #init = np.zeros(num_points)

    #df2["Pred Bg"] = np.zeros(init)
    #df2["Pred Bth"] = np.zeros(init)
    #df2["Pred Pe"] = np.zeros(init)
    #df2["phi_star"] = np.zeros(init)
    #df2["phi_th"] = np.zeros(init)
    #df2["Bth6"] = np.zeros(init)
    #df2["phi_star_star"] = np.zeros(init)
    #df2["param"] = np.zeros(init)
    #df2["g"] = np.zeros(init)
    #df2["lam_g"] = np.zeros(init)
    #df2["Nw/Ne"] = np.zeros(init)
    #df2["lam_eta_sp/Pe^2"] = np.zeros(init)

    #df_create = pd.DataFrame(
    #        {
    #            "Polymer": pd.Series(data=init,dtype="str"),
    #            "Solvent": pd.Series(data=init,dtype="str"),
    #            "Group": pd.Series(data=init,dtype="int"),
    #            "True Bg": pd.Series(data=init,dtype="float"),
    #            "Pred Bg": pd.Series(data=init,dtype="float"),
    #            "True Bth": pd.Series(data=init,dtype="float"),
    #            "Pred Bth": pd.Series(data=init,dtype="float"),
    #            "Pred Pe": pd.Series(data=init,dtype="float"),
    #            "Nw": pd.Series(data=init,dtype="float"),
    #            "phi_star": pd.Series(data=init,dtype="float"),
    #            "phi_th": pd.Series(data=init,dtype="float"),
    #            "Bth6": pd.Series(data=init,dtype="float"),
    #            "phi_star_star": pd.Series(data=init,dtype="float"),
    #            "param": pd.Series(data=init,dtype="float"),
    #            "g": pd.Series(data=init,dtype="float"),
    #            "lam_g": pd.Series(data=init,dtype="float"),
    #            "eta_sp": pd.Series(data=init,dtype="float"),
    #            "Nw/Ne": pd.Series(data=init,dtype="float"),
    #            "lam_eta_sp/Pe^2": pd.Series(data=init,dtype="float"),
    #         }
    #    )

    counter = 0

    for i in np.unique(df_table2['Group']):

        Bg = np.unique(df_table2[df_table2["Group"] == i]["Pred Bg"])[0]
        Bth = np.unique(df_table2[df_table2["Group"] == i]["Pred Bth"])[0]
        Pe = np.unique(df_table2[df_table2["Group"] == i]["Pred Pe"])[0]

        mask = df2.loc[:, "group"] == i
        df2.loc[mask, "Pred Bg"] = Bg
        df2.loc[mask, "Pred Bth"] = Bth
        df2.loc[mask, "Pred Pe"] = Pe

    #Bg_true = torch.tensor(df2["Bg"].values)
    #Bth_true = torch.tensor(df2["Bth"].values)


    Bg_true = torch.tensor(df2["Bg"].values)
    Bth_true = torch.tensor(df2["Bth"].values)
    Bg_t = torch.tensor(df2["Pred Bg"].values)
    Bth_t = torch.tensor(df2["Pred Bth"].values)

    Bg_t[Bg_true==0] = torch.nan
    Bth_t[Bth_true==0] = torch.nan

    Nw_t = torch.tensor(df2["Nw"].values)

    phi_star = torch.where(Bg_true > 0, Bg_t**3*Nw_t**(1-3*0.588), Bth_t**3*Nw_t**(-0.5))
    phi_t = torch.tensor(df2["phi"].values)

    #df2.loc[mask, "phi_star"] = phi_t.numpy()

    #df2.loc[mask, "phi_star"] = df2.loc[mask, "Bg"]**3*df2.loc[mask, "Nw"]**(1-3*0.588)
    #df2.loc[~mask, "phi_star"] = df2.loc[~mask, "Bth"]**3*df2.loc[~mask, "Nw"]**(-0.5)

    df2["phi_star"] = phi_star.numpy()


    phi_th_t = Bth_t**3*(Bg_t/Bth_t)**(1/(2*0.588-1))
    Bth6_t = Bth_t**6
    phi_star_star_t = Bth_t**4
    param_t = (Bth_t/Bg_t)**(1/(2*0.588-1))/Bth_t**3

    df2["phi_th"] = phi_th_t.numpy()
    df2["Bth6"] = Bth6_t.numpy()
    df2["phi_star_star"] = phi_star_star_t.numpy()
    df2["param"] = param_t.numpy()

    #df2.loc[mask, "phi_th"] = phi_th_t.numpy()
    #df2.loc[mask, "Bth6"] = Bth6_t.numpy()
    #df2.loc[mask, "phi_star_star"] = phi_star_star_t.numpy()
    #df2.loc[mask, "param"] = param_t.numpy()

    #df2.loc[mask, "phi_th"] = df2.loc[mask, "Pred Bth"]**3*(df2.loc[mask, "Pred Bg"]/df2.loc[mask, "Pred Bth"])**(1/(2*0.588-1))
    #df2.loc[mask, "Bth6"] = df2.loc[mask, "Pred Bth"]**6
    #df2.loc[mask, "phi_star_star"] = df2.loc[mask, "Pred Bth"]**4

    #df2.loc[mask, "param"] = (df2.loc[mask, "Pred Bth"] / df2.loc[mask, "Pred Bg"])**(1/(2*0.588-1))/df2.loc[mask, "Pred Bth"]**3
    
    #phi_star_star_t = torch.tensor(df2["phi_star_star"].values)
    #phi_th_t = torch.tensor(df2["phi_th"].values)
    #Bth_t = torch.tensor(df2["Pred Bth"].values)
    #Bg_t = torch.tensor(df2["Pred Bg"].values)
    #Bth6_t = torch.tensor(df2["Bth6"].values)

    #df2["g"] = np.nanmin(df2["Pred Bg"]**(3/0.764)/df2["phi"]**(1/0.764), df2["Pred Bth"]**6/df2["phi"]**2)
    g_t = torch.fmin(
            Bg_t**(3/0.764) / phi_t**(1/0.764),
            Bth_t**6 / phi_t**2
        )

    lam_g_KN = torch.where(phi_t<phi_star_star_t, 1, (phi_t/phi_star_star_t)**(-3/2))
    lam_g_RC = torch.where(phi_t<phi_th_t, phi_th_t**(2/3)*Bth_t**-4, torch.where(phi_t<Bth6_t, phi_t**(2/3)*Bth_t**-4, torch.where(phi_t<phi_star_star_t, 1, (phi_t/phi_star_star_t)**(-3/2))))

    lam_g_Bth = torch.where(phi_t<Bth6_t, phi_t**(2/3)*Bth_t**-4, torch.where(phi_t<phi_star_star_t, 1, (phi_t/phi_star_star_t)**(-3/2)))

    lam_KN = 1 / lam_g_KN * torch.where(phi_t < phi_star_star_t, 1, (phi_t/phi_star_star_t)**(-1/2))
    lam_RC = 1 / lam_g_RC * torch.where(phi_t < phi_star_star_t, 1, (phi_t/phi_star_star_t)**(-1/2))
    lam_Bth = 1 / lam_g_Bth * torch.where(phi_t < phi_star_star_t, 1, (phi_t/phi_star_star_t)**(-1/2))

    #param_t = torch.tensor(df2["param"].values)

    lam_g = torch.where(torch.isnan(param_t), lam_Bth, torch.where(param_t >= 1, lam_g_KN, lam_g_RC))
    lam_g = torch.where(Bth_t.isnan(), 1, lam_g)

    lam = 1 / lam_g * torch.where(phi_t < phi_star_star_t, 1, (phi_t/phi_star_star_t)**(-1/2))
    lam = torch.where(Bth_t.isnan(), 1, lam)

    df2["lam_g"] = lam_g.numpy()
    df2["lam"] = lam.numpy()

    #g_t = torch.tensor(df2["g"].values)
    Pe_t = torch.tensor(df2["Pred Pe"].values)
    eta_sp = torch.tensor(df2["eta_sp"].values)

    Ne = Pe_t**2 * g_t * lam_g

    Nw = torch.tensor(df2["Nw"].values)

    Nw_Ne = Nw/Ne

    lam_eta_sp_Pe2 = lam*eta_sp/Pe_t**2

    df2["Nw_Ne"] = Nw_Ne.numpy()
    df2["lam_eta_sp_Pe2"] = lam_eta_sp_Pe2

    mask = (df2.loc[:, "Bth"] > 0) & (df2.loc[:, "Bg"] > 0)

    df2.to_csv("df_universal_plot.csv")
    print(df2)
