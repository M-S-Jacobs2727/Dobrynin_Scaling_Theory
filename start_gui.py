# TODO: collect g, N/g, lamda_g, lamda, lamda*eta_sp into df

import re, sys, os, pickle
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, 
    NavigationToolbar2Tk
)
import matplotlib.pyplot as plt

plt.style.use('mikestyle')

import seaborn as sns
from scipy.optimize import least_squares
from scipy.optimize.zeros import brentq
from dataclasses import dataclass, field, asdict

import tkinter as tk
from tkinter import filedialog as fd
from tkinter import messagebox as mb
from tkinter import LEFT, RIGHT, END
from tkinter.ttk import Frame, Label, Entry, Button, Checkbutton, Spinbox

import scalingEqs

BG_COLOR = '#F0F0F0'

@dataclass(init=False)
class System:
    # Solution descriptors
    polymer: str
    solvent: str
    temperature: int
    salt_type: str
    salt_valence: int
    salt_concentration: float
    concentration_units: str
    N_units: str

    # Data
    datafile: str = ''

    # Scaling parameters
    Bpe: float = 0
    Bg: float = 0
    Bth: float = 0
    cD: float = 0
    cth: float = 0
    c88: float = 0
    bK: float = 0
    v: float = 0

class AnalysisApp:

    def __init__(self, master):
        # Initialize window settings
        self.master = master
        self.master.title("Viscosity Data Scaling Analysis")
        self.master.geometry('1500x1000+0+0') # default size if 'restored'
        self.master.state('zoomed')
        self.master.configure(bg=BG_COLOR)

        # TODO: find a way to get window size in pixels instead of hard-coding
        win_width = 2560
        win_height = 1400

        # Set up frames on a grid. Left frame is for normalized viscosity plots
        # Right frame is split into main universal plot (visc v N/g) on top, 
        # and a parameter tweaking interface below. Both left and right frames
        # have the matplotlib toolbar in the bottom 5% of the frame.
        self.master.columnconfigure(0, weight=1)
        self.master.columnconfigure(1, weight=1)
        self.master.rowconfigure(0, weight=80)
        self.master.rowconfigure(1, weight=15)
        self.master.rowconfigure(2, weight=5)

        self.leftFrame = Frame(self.master)
        self.leftFrame.grid(column=0, row=0, rowspan=2, sticky=tk.NSEW)

        self.rTopFrame = Frame(self.master)
        self.rTopFrame.grid(column=1, row=0, sticky=[tk.N, tk.E, tk.W])

        self.rBotFrame = Frame(self.master, borderwidth=3, relief='groove', takefocus=True)
        self.rBotFrame.grid(column=1, row=1, sticky=[tk.N, tk.E, tk.W], padx=8)

        self.lBarFrame = Frame(self.master)
        self.lBarFrame.grid(column=0, row=2, sticky=tk.W)

        self.rBarFrame = Frame(self.master)
        self.rBarFrame.grid(column=1, row=2, sticky=tk.W)

        # Left frame: normalized plots with plateaus
        # TODO: again, find a better solution for figure size.
        dpi = 100

        norm_fig = Figure(
            figsize=(win_width/dpi/2, win_height/dpi), 
            dpi=dpi
        )
        norm_fig.patch.set_facecolor(BG_COLOR)
        self.norm_axes = norm_fig.subplots(
            3, 2, 
            subplot_kw={'xscale':'log', 'yscale':'log'}
        )
        norm_fig.subplots_adjust(
            left=0.1,
            right=0.95,
            bottom=0.1,
            top=0.95,
            hspace=0.25,
            wspace=0.3
        )

        self.leftCanvas = FigureCanvasTkAgg(norm_fig, master=self.leftFrame)  
        self.leftCanvas.draw()

        self.leftCanvas.get_tk_widget().pack()
    
        leftToolbar = NavigationToolbar2Tk(self.leftCanvas, self.lBarFrame)
        leftToolbar.update()

        self.leftCanvas.get_tk_widget().pack()

        # Right frame: final plot and parameter textboxes
        # TODO: again, find a better solution for figure size.
        finalFig = Figure(
            figsize=(win_width/dpi/2, win_height/dpi*0.80), 
            dpi=dpi
        )
        finalFig.patch.set_facecolor(BG_COLOR)
        self.finalAx = finalFig.add_subplot(
            111, 
            xscale='log',
            yscale='log'
        )
        finalFig.subplots_adjust(top=0.95, bottom=0.15)
        
        self.rightCanvas = FigureCanvasTkAgg(finalFig, master=self.rTopFrame)  
        self.rightCanvas.draw()

        self.rightCanvas.get_tk_widget().pack()
    
        rightToolbar = NavigationToolbar2Tk(self.rightCanvas, self.rBarFrame)
        rightToolbar.update()

        self.rightCanvas.get_tk_widget().pack()

        # Right frame, bottom area for parameter tweaking. 
        # TODO: figure out how to spread out rows, make it look nice.
        self.rBotFrame.columnconfigure(0, weight=1)
        self.rBotFrame.columnconfigure(1, weight=4)
        self.rBotFrame.columnconfigure(2, weight=1)
        self.rBotFrame.columnconfigure(3, weight=4)
        self.rBotFrame.columnconfigure(4, weight=1)
        self.rBotFrame.columnconfigure(5, weight=4)
        self.rBotFrame.columnconfigure(6, weight=1)
        self.rBotFrame.columnconfigure(7, weight=4)

        self.rBotFrame.rowconfigure(0, weight=1)
        self.rBotFrame.rowconfigure(1, weight=1)
        self.rBotFrame.rowconfigure(2, weight=1)
        self.rBotFrame.rowconfigure(3, weight=1)
        self.rBotFrame.rowconfigure(4, weight=2)

        # Checkboxes
        self.Bpe_on = tk.BooleanVar(value=False)
        self.Bg_on = tk.BooleanVar(value=True)
        self.Bth_on = tk.BooleanVar(value=True)
        self.Pe_on = tk.BooleanVar(value=True)

        Bpe_check = Checkbutton(
            self.rBotFrame, 
            variable=self.Bpe_on, 
            onvalue=True, 
            width=0,
            command=self.onToggleBpe
        )
        Bg_check = Checkbutton(
            self.rBotFrame, 
            variable=self.Bg_on, 
            onvalue=True, 
            width=0,
            command=self.onToggleBg
        )        
        Bth_check = Checkbutton(
            self.rBotFrame, 
            variable=self.Bth_on, 
            onvalue=True, 
            width=0,
            command=self.onToggleBth
        )
        Pe_check = Checkbutton(
            self.rBotFrame, 
            variable=self.Pe_on, 
            onvalue=True, 
            width=0,
            command=self.onTogglePe
        )

        Bpe_check.grid(column=1, row=0, sticky=tk.NW, pady=10)
        Bg_check.grid(column=3, row=0, sticky=tk.NW, pady=10)
        Bth_check.grid(column=5, row=0, sticky=tk.NW, pady=10)
        Pe_check.grid(column=7, row=0, sticky=tk.NW, pady=10)

        # Parameter labels and entry fields. Make a list/loop structure?
        Label(
            self.rBotFrame, 
            text='Bpe', 
            width=6, 
            justify='right',
            font=16
        ).grid(row=1, column=0, padx=2, sticky=tk.E, pady=10)
        self.Bpe_spinbox = Spinbox(
            self.rBotFrame, 
            width=10, 
            font=16, 
            from_=1,
            to=20,
            increment=0.1,
            state='disabled'
        )
        self.Bpe_spinbox.grid(row=1, column=1, padx=2, sticky=tk.W, pady=10)
        self.Bpe_spinbox.configure(state='disabled')

        self.cD_value = tk.StringVar()
        Label(
            self.rBotFrame, 
            text='cD', 
            width=6, 
            justify='right',
            font=16
        ).grid(row=2, column=0, padx=2, sticky=tk.E, pady=10)
        Label(
            self.rBotFrame, 
            width=10, 
            font=16, 
            textvariable=self.cD_value
        ).grid(row=2, column=1, padx=2, sticky=tk.W, pady=10)

        Label(
            self.rBotFrame, 
            text='Bg', 
            width=6, 
            justify='right',
            font=16
        ).grid(row=1, column=2, padx=2, sticky=tk.E, pady=10)
        self.Bg_spinbox = Spinbox(
            self.rBotFrame, 
            width=10, 
            font=16, 
            from_=0,
            to=2,
            increment=0.05,
            state='disabled'
        )
        self.Bg_spinbox.grid(row=1, column=3, padx=2, sticky=tk.W, pady=10)

        self.cth_value = tk.StringVar()
        Label(
            self.rBotFrame, 
            text='cth', 
            width=6, 
            justify='right',
            font=16
        ).grid(row=2, column=2, padx=2, sticky=tk.E, pady=10)
        Label(
            self.rBotFrame, 
            width=10, 
            font=16, 
            textvariable=self.cth_value
        ).grid(row=2, column=3, padx=2, sticky=tk.W, pady=10)
        
        Label(
            self.rBotFrame, 
            text='Bth', 
            width=6, 
            justify='right',
            font=16
        ).grid(row=1, column=4, padx=2, sticky=tk.E, pady=10)
        self.Bth_spinbox = Spinbox(
            self.rBotFrame, 
            width=10, 
            font=16, 
            from_=0,
            to=1,
            increment=0.01,
            state='disabled'
        )
        self.Bth_spinbox.grid(row=1, column=5, padx=2, sticky=tk.W, pady=10)

        self.c88_value = tk.StringVar()
        Label(
            self.rBotFrame, 
            text='c**', 
            width=6, 
            justify='right',
            font=16
        ).grid(row=2, column=4, padx=2, sticky=tk.E, pady=10)
        Label(
            self.rBotFrame, 
            width=10, 
            font=16, 
            textvariable=self.c88_value
        ).grid(row=2, column=5, padx=2, sticky=tk.W, pady=10)

        Label(
            self.rBotFrame, 
            text='Pe', 
            width=6, 
            justify='right',
            font=16
        ).grid(row=1, column=6, padx=2, sticky=tk.E, pady=10)
        self.Pe_spinbox = Spinbox(
            self.rBotFrame,
            width=10,
            font=16,
            from_=1,
            to=100,
            increment=0.1,
            state='disabled'
        )
        self.Pe_spinbox.grid(row=1, column=7, padx=2, sticky=tk.W, pady=10)

        self.initialize_btn = Button(
            self.rBotFrame, 
            text='Initialize', 
            underline=0, 
            width=16, 
            command=self.onInitialize,
            state='disabled'
        )
        self.initialize_btn.grid(row=3, column=1, pady=10, sticky=tk.W)
        
        self.recalculate_btn = Button(
            self.rBotFrame, 
            text='Recalculate', 
            underline=0, 
            width=16, 
            command=self.onRecalculate,
            state='disabled'
        )
        self.recalculate_btn.grid(row=3, column=3, pady=10, sticky=tk.W)
        
        self.refit_btn = Button(
            self.rBotFrame, 
            text='Fit Pe', 
            underline=0, 
            width=16, 
            command=self.onRefit,
            state='disabled'
        )
        self.refit_btn.grid(row=3, column=5, pady=10, sticky=tk.W)
        
        self.save_btn = Button(
            self.rBotFrame, 
            text='Save', 
            underline=0, 
            width=16,
            command=self.onSave,
            state='disabled'
        )
        self.save_btn.grid(row=3, column=7, pady=10, sticky=tk.W)

        # File loading interactives
        fileFrame = Frame(self.rBotFrame)
        fileFrame.grid(row=4, column=0, columnspan=8, sticky=tk.NSEW)

        Label(
            fileFrame, 
            text='Select File', 
            width=10, 
            font=16
        ).pack(side=LEFT, padx=5, pady=10)

        self.load_btn = Button(
            fileFrame,
            text='Load',
            width=12,
            command=self.onLoad
        )
        self.load_btn.pack(padx=5, pady=10, side=RIGHT)

        self.browse_btn = Button(
            fileFrame, 
            text='Browse...', 
            width=12, 
            command=self.onBrowse
        )
        self.browse_btn.pack(padx=5, pady=10, side=RIGHT)
        self.browse_btn.focus_force()

        self.filenameEntry = Entry(fileFrame, font=12)
        self.filenameEntry.pack(fill=tk.X, padx=5, expand=True)

    def onBrowse(self):
        # Execeuted with the 'Browse...' button
        self.filepath = fd.askopenfilename(
            initialdir='CSV_files', 
            filetypes=(('CSV Files', '*.csv'),)
        )
        if self.filepath != '':
            self.load_btn.focus()
            self.filenameEntry.delete(0, END)
            self.filenameEntry.insert(0, self.filepath)

    def onLoad(self):
        # Load data from file
        self.filepath = self.filenameEntry.get()
        self.df = pd.read_csv(self.filepath)
        self.df = self.expandDataframe()

        # Populate plots in both frames
        self.plotNormalizedViscosity()
        self.leftCanvas.draw()

        self.plotOriginalViscosity()
        self.rightCanvas.draw()

        # Enabled analysis
        self.initialize_btn.configure(state='active')
        self.initialize_btn.focus()
    
    def plotNormalizedViscosity(self):
        for i in range(3):
            for j in range(2):
                self.norm_axes[i, j].clear()
                self.norm_axes[i, j].set_xscale('log')
                self.norm_axes[i, j].set_yscale('log')

        sns.scatterplot(
            data = self.df, 
            x='c', 
            y = 'visc_norm_0.5', 
            hue='N', 
            palette='tab10',
            legend=False,
            ax=self.norm_axes[0, 0]
        )
        
        sns.scatterplot(
            data = self.df, 
            x='c', 
            y = 'visc_norm_1.5', 
            hue='N', 
            palette='tab10',
            legend=False,
            ax=self.norm_axes[0, 1]
        )
        
        sns.scatterplot(
            data = self.df, 
            x='c', 
            y = 'visc_norm_1.31', 
            hue='N', 
            palette='tab10',
            legend=False,
            ax=self.norm_axes[1, 0]
        )
        
        sns.scatterplot(
            data = self.df, 
            x='c', 
            y = 'visc_norm_3.93', 
            hue='N', 
            palette='tab10',
            legend=False,
            ax=self.norm_axes[1, 1]
        )
        
        sns.scatterplot(
            data = self.df, 
            x='c', 
            y = 'visc_norm_2', 
            hue='N', 
            palette='tab10',
            legend=False,
            ax=self.norm_axes[2, 0]
        )
        
        sns.scatterplot(
            data = self.df, 
            x='c', 
            y = 'visc_norm_6', 
            hue='N', 
            palette='tab10',
            legend=False,
            ax=self.norm_axes[2, 1]
        )

    def plotOriginalViscosity(self):
        self.finalAx.clear()
        self.finalAx.set_xscale('log')
        self.finalAx.set_yscale('log')
        sns.scatterplot(
            data = self.df, 
            x='c', 
            y = 'visc', 
            hue='N', 
            palette='tab10',
            s=160,
            ax=self.finalAx
        )
        self.finalAx.set_xlabel(r'$\mathit{cl^3}$', fontsize=36)
        self.finalAx.set_ylabel(r'$\mathit{\eta_{sp}}$', fontsize=36)

    def expandDataframe(self):
        self.df[f'visc_norm_0.5'] = (
            self.df['visc'] / self.df['N'] / self.df['c']**0.5
        )
        self.df[f'visc_norm_1.5'] = (
            self.df['visc'] / self.df['N'] / self.df['c']**1.5
        )
        self.df[f'visc_norm_1.31'] = (
            self.df['visc'] / self.df['N'] / self.df['c']**(1/(3*0.588-1))
        )
        self.df[f'visc_norm_3.93'] = (
            self.df['visc'] / self.df['N'] / self.df['c']**(3/(3*0.588-1))
        )
        self.df[f'visc_norm_2'] = (
            self.df['visc'] / self.df['N'] / self.df['c']**2
        )
        self.df[f'visc_norm_6'] = (
            self.df['visc'] / self.df['N'] / self.df['c']**6
        )
        self.df['N'] = np.around(self.df['N'])
        self.df = self.df.astype({'N':int})
        return self.df

    def getCrossoverConcentrations(self, csfz=0):
        if csfz:
            cD = brentq(
                lambda c: (
                    self.Bg**2 
                    - c**0.412 * (self.Bpe * np.sqrt(1 + 2*csfz/c))**0.764
                ),
                0, 100
            )
        else:
            cD = self.Bpe**3 * (self.Bg / self.Bpe)**(1/0.206)
        cth = self.Bth**3 * (self.Bth / self.Bg)**(1/0.176)
        c88 = self.Bth**4
        b_l = self.Bth**-2
        return cD, cth, c88, b_l

    def correctForRC(self):
        cthb3 = self.cth * self.b_l**3
        c = self.df['c']
        lamda_g = np.ones_like(c)
        if cthb3 > 0.9:
            return lamda_g
        lamda_g[c>self.cth] = (c[c>self.cth]/self.cth)**(2/3)
        lamda_g[c>self.b_l**-3] = cthb3**(-2/3)
        return lamda_g

    def correctForConcentrated(self, lamda_g):
        lamda = 1 / lamda_g
        c = self.df['c']
        lamda[c>self.c88] = c[c>self.c88] / self.c88
        return lamda

    def DPinCorrBlob(self):
        c = self.df['c']
        g_pe = np.sqrt(self.Bpe**3/c)
        g_good = (self.Bg**3/c)**(1/0.764)
        g_th = (self.Bth**3/c)**2
        g = np.amin(np.stack((g_pe, g_good, g_th), axis=0), axis=0)
        return g

    def onToggleBpe(self):
        if self.Bpe_on.get():
            self.Bpe_spinbox.configure(state='active')
            self.recalculate_btn.configure(state='active')
            self.refit_btn.configure(state='active')
        elif (not self.Bg_on) and (not self.Bth_on):
            self.recalculate_btn.configure(state='disabled')
            self.refit_btn.configure(state='disabled')
        else:
            self.Bpe_spinbox.configure(state='disabled')

    def onToggleBg(self):
        if self.Bg_on.get():
            self.Bg_spinbox.configure(state='active')
            self.recalculate_btn.configure(state='active')
            self.refit_btn.configure(state='active')
        elif (not self.Bpe_on) and (not self.Bth_on):
            self.recalculate_btn.configure(state='disabled')
            self.refit_btn.configure(state='disabled')
        else:
            self.Bg_spinbox.configure(state='disabled')

    def onToggleBth(self):
        if self.Bth_on.get():
            self.Bth_spinbox.configure(state='active')
            self.recalculate_btn.configure(state='active')
            self.refit_btn.configure(state='active')
        elif (not self.Bg_on) and (not self.Bpe_on):
            self.recalculate_btn.configure(state='disabled')
            self.refit_btn.configure(state='disabled')
        else:
            self.Bth_spinbox.configure(state='disabled')

    def onTogglePe(self):
        if self.Pe_on.get():
            #self.Pe_spinbox.configure(state='active')
            self.Pe_spinbox.configure(state='active')
        else:
            #self.Pe_spinbox.configure(state='disabled')
            self.Pe_spinbox.configure(state='disabled')

    def onInitialize(self):
        self.Bpe = self.Bg = self.Bth = 1e9
        if not (self.Bpe_on.get() or self.Bg_on.get() or self.Bth_on.get()):
            return

        # Calculate initial plateaus based on minima, use as initial guess p0
        if self.Bpe_on.get():
            self.Bpe = min(self.df['visc_norm_0.5'])**(-2/3)
        if self.Bg_on.get():
            self.Bg = min(self.df['visc_norm_1.31'])**(1/3-0.588)
        if self.Bth_on.get():
            self.Bth = min(self.df['visc_norm_2'])**(-1/6)

        p0 = [self.Bpe, self.Bg, self.Bth, 10]
        
        # Fit the big function
        res = least_squares(
            lambda p: fitFunction(
                p, 
                self.df['c'], 
                self.df['N']
            ) - self.df['visc'],
            p0,
            method='trf',
            jac='3-point',
            bounds=((0, 0, 0, 2), 
                    (1e10, 1e10, 1e10, 100)
            ),
            xtol=1e-10,
            verbose=0,
            loss='cauchy'
        )
        self.Bpe, self.Bg, self.Bth, self.Pe = res.x
        (self.cD, self.cth, 
                self.c88, self.b_l) = self.getCrossoverConcentrations()

        # TODO: If pe regime exists, check for residual salt in c < cD

        # If Rubinstein-Colby parameter indicates marginally good solvent, 
        # or if c > c** exists, then recalculate with lambda
        lamda_g = self.correctForRC()
        lamda = self.correctForConcentrated(lamda_g)
        g = self.DPinCorrBlob() * lamda_g

        # Then refit Pe only
        res = least_squares(
            lambda pe: viscFunc1(self.df['N'], pe, g) - lamda*self.df['visc'],
            self.Pe,
            method='trf',
            jac='3-point',
            bounds=(0, 100),
            xtol=1e-10,
            verbose=0,
            loss='cauchy'
        )
        self.Pe, = res.x

        # Draw plateaus and arrows on left figure
        if self.Bpe_on.get():
            Bpe_plat = self.Bpe**-1.5
            self.Bpe_spinbox.configure(state='active')
            self.Bpe_spinbox.delete(0, END)
            self.Bpe_spinbox.insert(0, f'{self.Bpe:.2f}')
            self.cD_value.set(f'{self.cD:.6f}')
            xlo, xhi = self.norm_axes[0, 0].get_xlim()
            ylo, yhi = self.norm_axes[0, 0].get_ylim()
            self.Bpe_plat_line, = self.norm_axes[0, 0].plot(
                [xlo, xhi], 
                [Bpe_plat, Bpe_plat], 
                'k--'
            )
            shift_up = (yhi/ylo)**0.05
            self.cD_arrow_1, = self.norm_axes[0, 0].plot(
                self.cD, ylo*shift_up, 'ko', 
                marker=r'$\downarrow$', markersize=20
            )
            self.norm_axes[0, 0].set_xlim((xlo, xhi))
            self.norm_axes[0, 0].set_ylim((ylo, yhi))
        if self.Bg_on.get():
            Bg_plat = self.Bg**(-3/(3*0.588-1))
            self.Bg_spinbox.configure(state='active')
            self.Bg_spinbox.delete(0, END)
            self.Bg_spinbox.insert(0, f'{self.Bg:.2f}')
            self.cth_value.set(f'{self.cth:.6f}')
            xlo, xhi = self.norm_axes[1, 0].get_xlim()
            ylo, yhi = self.norm_axes[1, 0].get_ylim()
            self.Bg_plat_line, = self.norm_axes[1, 0].plot(
                [xlo, xhi], 
                [Bg_plat, Bg_plat], 
                'k--'
            )
            shift_up = (yhi/ylo)**0.05
            self.cD_arrow_2, = self.norm_axes[1, 0].plot(
                self.cD, ylo*shift_up, 'ko', 
                marker=r'$\downarrow$', markersize=20
            )
            self.cth_arrow_1, = self.norm_axes[1, 0].plot(
                self.cth, ylo*shift_up, 'ko', 
                marker=r'$\downarrow$', markersize=20
            )
            self.norm_axes[1, 0].set_xlim((xlo, xhi))
            self.norm_axes[1, 0].set_ylim((ylo, yhi))
        if self.Bth_on.get():
            Bth_plat = self.Bth**(-6)
            self.Bth_spinbox.configure(state='active')
            self.Bth_spinbox.delete(0, END)
            self.Bth_spinbox.insert(0, f'{self.Bth:.2f}')
            self.c88_value.set(f'{self.c88:.6f}')
            xlo, xhi = self.norm_axes[2, 0].get_xlim()
            ylo, yhi = self.norm_axes[2, 0].get_ylim()
            self.Bth_plat_line, = self.norm_axes[2, 0].plot(
                [xlo, xhi], 
                [Bth_plat, Bth_plat], 
                'k--'
            )
            shift_up = (yhi/ylo)**0.05
            self.cth_arrow_2, = self.norm_axes[2, 0].plot(
                self.cth, ylo*shift_up, 'ko', 
                marker=r'$\downarrow$', markersize=20
            )
            self.c88_arrow_1, = self.norm_axes[2, 0].plot(
                self.c88, ylo*shift_up, 'ko', 
                marker=r'$\downarrow$', markersize=20
            )
            self.norm_axes[2, 0].set_xlim((xlo, xhi))
            self.norm_axes[2, 0].set_ylim((ylo, yhi))
        self.Pe_spinbox.configure(state='active')
        self.Pe_spinbox.delete(0, END)
        self.Pe_spinbox.insert(0, f'{self.Pe:.2f}')

        self.leftCanvas.draw()

        # Calculate N/g, redraw right figure
        self.df['N/g'] = self.df['N'] / g
        self.df['lamda*visc'] = lamda * self.df['visc']
        self.finalAx.clear()

        sns.scatterplot(
            data = self.df, 
            x='N/g', 
            y = 'lamda*visc', 
            hue='N', 
            palette='tab10',
            ax=self.finalAx,
            s=160
        )
        self.finalAx.set_xscale('log')
        self.finalAx.set_yscale('log')
            
        x_geom = np.geomspace(1, max(self.df['N']/g), 100)
        self.fit_curve, = self.finalAx.plot(
            x_geom, viscFunc1(x_geom, self.Pe, 1), 'k-'
        )
        if (lamda_g == 1).all():
            self.finalAx.set_xlabel(r'$\mathit{N/g}$', fontsize=36)
        else:
            self.finalAx.set_xlabel(r'$\mathit{N/\tilde g}$', fontsize=36)
        if (lamda == 1).all():
            self.finalAx.set_ylabel(r'$\mathit{\eta_{sp}}$', fontsize=36)
        else:
            self.finalAx.set_ylabel(
                r'$\mathit{\lambda\eta_{sp}}$', 
                fontsize=36
            )

        self.rightCanvas.draw()

        # Set up next buttons
        self.recalculate_btn.configure(state='active')
        self.refit_btn.configure(state='active')
        self.save_btn.configure(state='active')
        self.save_btn.focus()

    def onRecalculate(self):
        # TODO: make this logic
        # Only allow one value to change at a time. If the user changes 
        # multiple parameters, reset them to the last state and warn them.
        changes = []
        if self.Bpe_on.get() and self.Bpe_spinbox.get() != f'{self.Bpe:.2f}':
            changes.append(0)
        if self.Bg_on.get() and self.Bg_spinbox.get() != f'{self.Bg:.2f}':
            changes.append(1)
        if self.Bth_on.get() and self.Bth_spinbox.get() != f'{self.Bth:.2f}':
            changes.append(2)
        if self.Pe_on.get() and self.Pe_spinbox.get() != f'{self.Pe:.2f}':
            changes.append(3)

        print(changes)
        if len(changes) == 0:
            return
        if len(changes) > 1:
            if self.Bpe_on.get():
                self.Bpe_spinbox.delete(0, END)
                self.Bpe_spinbox.insert(0, f'{self.Bpe:.2f}')
            if self.Bg_on.get():
                self.Bg_spinbox.delete(0, END)
                self.Bg_spinbox.insert(0, f'{self.Bg:.2f}')
            if self.Bth_on.get():
                self.Bth_spinbox.delete(0, END)
                self.Bth_spinbox.insert(0, f'{self.Bth:.2f}')
            if self.Pe_on.get():
                self.Pe_spinbox.delete(0, END)
                self.Pe_spinbox.insert(0, f'{self.Pe:.2f}')
            return

        # Get the values that have changed and recalculate them
        change = changes[0]
        if change == 0:
            self.Bpe = float(self.Bpe_spinbox.get())
            self.cD = scalingEqs.get_cD(Bpe=self.Bpe, Bg=self.Bg)
            self.cD_value.set(f'{self.cD:.6f}')
        elif change == 1:
            self.Bg = float(self.Bg_spinbox.get())
            if self.Bpe_on.get():
                self.cD = scalingEqs.get_cD(Bpe=self.Bpe, Bg=self.Bg)
                self.cD_value.set(f'{self.cD:.6f}')
            if self.Bth_on.get():
                self.cth = scalingEqs.get_cth(Bth=self.Bth, Bg=self.Bg)
                self.cth_value.set(f'{self.cth:.6f}')
        elif change == 2:
            self.Bth = float(self.Bth_spinbox.get())
            if self.Bg_on.get():
                self.cth = scalingEqs.get_cth(Bth=self.Bth, Bg=self.Bg)
                self.cth_value.set(f'{self.cth:.6f}')
            self.c88 = scalingEqs.get_c88(Bth=self.Bth)
            self.c88_value.set(f'{self.c88:.6f}')
        elif change == 3:
            self.Pe = float(self.Pe_spinbox.get())

        # TODO: If pe regime exists, check for residual salt in c < cD

        # If Rubinstein-Colby parameter indicates marginally good solvent, 
        # or if c > c** exists, then recalculate with lambda
        lamda_g = self.correctForRC()
        lamda = self.correctForConcentrated(lamda_g)
        g = self.DPinCorrBlob() * lamda_g

        if change != 3:

            # Then refit Pe only
            res = least_squares(
                lambda pe: viscFunc1(self.df['N'], pe, g) - lamda*self.df['visc'],
                self.Pe,
                method='trf',
                jac='3-point',
                bounds=(0, 100),
                xtol=1e-10,
                verbose=0,
                loss='cauchy'
            )
            self.Pe, = res.x

            # Draw plateaus and arrows on left figure
            if self.Bpe_on.get():
                Bpe_plat = self.Bpe**-1.5
                self.Bpe_plat_line.set_ydata([Bpe_plat, Bpe_plat])
                self.cD_arrow_1.set_xdata(self.cD)
            if self.Bg_on.get():
                Bg_plat = self.Bg**(-3/(3*0.588-1))
                self.Bg_plat_line.set_ydata([Bg_plat, Bg_plat])
                self.cD_arrow_2.set_xdata(self.cD)
                self.cth_arrow_1.set_xdata(self.cth)
            if self.Bth_on.get():
                Bth_plat = self.Bth**(-6)
                self.Bth_plat_line.set_ydata([Bth_plat, Bth_plat])
                self.cth_arrow_2.set_xdata(self.cth)
                self.c88_arrow_1.set_xdata(self.c88)
            self.Pe_spinbox.delete(0, END)
            self.Pe_spinbox.insert(0, f'{self.Pe:.2f}')

            self.leftCanvas.draw()

            # Recalculate N/g and Pe, redraw right figure
            self.df['N/g'] = self.df['N'] / g
            self.df['lamda*visc'] = lamda * self.df['visc']
            self.finalAx.clear()

            sns.scatterplot(
                data = self.df, 
                x='N/g', 
                y = 'lamda*visc', 
                hue='N', 
                palette='tab10',
                ax=self.finalAx,
                s=160
            )
            self.finalAx.set_xscale('log')
            self.finalAx.set_yscale('log')
            
            x_geom = np.geomspace(1, max(self.df['N']/g), 100)
        
            self.fit_curve, = self.finalAx.plot(
                x_geom, viscFunc1(x_geom, self.Pe, 1), 'k-'
            )
        self.fit_curve.set_data(x_geom, viscFunc1(x_geom, self.Pe, 1))

        if (lamda_g == 1).all():
            self.finalAx.set_xlabel(r'$\mathit{N/g}$', fontsize=36)
        else:
            self.finalAx.set_xlabel(r'$\mathit{N/\tilde g}$', fontsize=36)
        if (lamda == 1).all():
            self.finalAx.set_ylabel(r'$\mathit{\eta_{sp}}$', fontsize=36)
        else:
            self.finalAx.set_ylabel(
                r'$\mathit{\lambda\eta_{sp}}$', fontsize=36
            )

        self.rightCanvas.draw()

    def onRefit(self):
        # TODO: If pe regime exists, check for residual salt in c < cD

        # If Rubinstein-Colby parameter indicates marginally good solvent, 
        # or if c > c** exists, then recalculate with lambda
        lamda_g = self.correctForRC()
        lamda = self.correctForConcentrated(lamda_g)
        g = self.DPinCorrBlob() * lamda_g

        # Then refit Pe only
        res = least_squares(
            lambda pe: viscFunc1(self.df['N'], pe, g) - lamda*self.df['visc'],
            self.Pe,
            method='trf',
            jac='3-point',
            bounds=(0, 100),
            xtol=1e-10,
            verbose=0,
            loss='cauchy'
        )
        self.Pe, = res.x
        self.Pe_spinbox.delete(0, END)
        self.Pe_spinbox.insert(0, f'{self.Pe:.2f}')

        x_geom = np.geomspace(1, max(self.df['N']/g), 100)
        self.fit_curve.set_data(x_geom, viscFunc1(x_geom, self.Pe, 1))
        if (lamda_g == 1).all():
            self.finalAx.set_xlabel(r'$\mathit{N/g}$', fontsize=36)
        else:
            self.finalAx.set_xlabel(r'$\mathit{N/\tilde g}$', fontsize=36)
        if (lamda == 1).all():
            self.finalAx.set_ylabel(r'$\mathit{\eta_{sp}}$', fontsize=36)
        else:
            self.finalAx.set_ylabel(
                r'$\mathit{\lambda\eta_{sp}}$', 
                fontsize=36
            )

        self.rightCanvas.draw()

    # TODO: build this with JSON? link to InfoDialog
    def onSave(self):
        return

# TODO: link to AnalysisApp.save(). Add M0 and l fields? (Maybe ask before
# for dimensional analysis)
# When adding fields, add to self.entryNames, self.processData, 
# self.initial_entries, and to the onSubmit() function.
class InfoDialog:

    def __init__(self, master):
        super().__init__()
        # self allow the variable to be used anywhere in the class
        self.master = master
        self.entryNames = [
            'Polymer Name', 'Solvent Name', 'Temperature [K]', 'Salt Type', 
            'Salt Valence', 'Salt Conc.', 'Conc. Units', 'N Units'
        ]
        self.processData = [
            lambda x: x.lower(), lambda x: x.lower(), lambda x: x.lower(), 
            float, int, float, lambda x: x, lambda x: x
        ]
        self.initial_entries = ['', '', '298','None', '1', '0', '', '']
        self.output = 0
        self.initUI()

    def initUI(self):

        self.frame = Frame(self.master)
        self.master.title("System Information")
        self.frame.pack(fill=tk.BOTH, expand=True)
        self.master.geometry(f'500x{len(self.entryNames)*40+50}')

        self.frames = []
        self.lbls = []
        self.entries = []

        for i in range(len(self.entryNames)):
            self.frames.append(Frame(self.frame))
            self.frames[i].pack(fill=tk.X)

            self.lbls.append(Label(
                self.frames[i], 
                text=self.entryNames[i], 
                width=16
            ))
            self.lbls[i].pack(side=LEFT, padx=5, pady=10)

            self.entries.append(Entry(self.frames[i]))
            self.entries[i].pack(fill=tk.X, padx=5, expand=True)
            self.entries[i].insert(0, self.initial_entries[i])

        self.lastFrame = Frame(self.frame)
        self.lastFrame.pack(fill=tk.X)

        # Command tells the form what to do when the button is clicked
        self.btn = Button(self.lastFrame, text="Submit", command=self.onSubmit)
        self.btn.pack(padx=5, pady=10)

        self.entries[0].focus()

    def onSubmit(self):
        system = System()
        system.polymer = self.entries[0].get().lower()
        system.solvent = self.entries[1].get().lower()
        system.temperature = int(self.entries[2].get())
        system.salt_type = self.entries[3].get()
        system.salt_valence = int(self.entries[4].get())
        system.salt_concentration = float(self.entries[5].get())
        system.concentration_units = self.entries[6].get()
        system.N_units = self.entries[7].get()

        self.output = system
        
        self.master.quit()

# Deprecated, program immediately starts with analysis app
class MainDialog:

    def __init__(self, master):
        super().__init__()
        # self allow the variable to be used anywhere in the class
        self.master = master
        self.frame = Frame(self.master)
        self.filepath = ''
        self.initUI()

    def initUI(self):

        self.master.title("Viscosity Scaling GUI")
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        # First line: file selection
        self.frame1 = Frame(self.frame)
        self.frame1.pack(fill=tk.X)
        self.label1 = Label(self.frame1, text='Select File', width=10)
        self.label1.pack(side=LEFT, padx=5, pady=10)
        self.browse_btn = Button(
            self.frame1, 
            text='Browse...', 
            width=12, 
            command=self.onBrowse
        )
        self.browse_btn.pack(padx=5, pady=10, side=RIGHT)
        self.entry1 = Entry(self.frame1, textvariable=self.filepath)
        self.entry1.pack(fill=tk.X, padx=5, expand=True)
            
        # Second line: Analyze Button
        frame2 = Frame(self.frame)
        frame2.pack(fill=tk.X)
        analyze_btn = Button(
            frame2, 
            text='Analyze', 
            width=10, 
            command=self.onAnalyze
        )
        analyze_btn.pack(padx=5, pady=10)

    def onAnalyze(self):
        runAnalysis(self.master, self.filepath)

    def onBrowse(self):
        # Execeuted with the 'Browse...' button
        self.filepath = fd.askopenfilename(
            initialdir='CSV_files', 
            filetypes=(('CSV Files', '*.csv'),)
        )
        if self.filepath != '':
            self.entry1.delete(0, END)
            self.entry1.insert(0, self.filepath)

def viscFunc1(N, Pe, g):
    return N / g * (1 + N**2 / Pe**4 / g**2)

def viscFunc2(N, Pe, g):
    return N / g * (1 + N / Pe**2 / g)**2

def fitFunction(params, c, N):
    """Crossover viscosity function between Rouse (unentangled) and entangled
    regimes, through different regimes reflecting fractal chain statistics.
    """
    Bpe, Bg, Bth, Pe = params
    g = DPinCorrBlob(Bpe, Bg, Bth, c)
    visc = viscFunc1(N, Pe, g)
    return visc

# Deprecated
def DPinCorrBlob(Bpe, Bg, Bth, c, return_regimes=False):
    g_pe = np.sqrt(Bpe**3/c)
    g_good = (Bg**3/c)**(1/0.764)
    g_th = (Bth**3/c)**2
    g = np.amin(np.stack((g_pe, g_good, g_th), axis=0), axis=0)
    if return_regimes:
        regimes = np.argmin(np.stack((g_pe, g_good, g_th), axis=0), axis=0)
        return g, regimes
    else:
        return g

# Deprecated
def correctForRC(c, cth, b_l):
    cthb3 = cth * b_l**3
    lamda_g = np.ones_like(c)
    if cthb3 > 0.9:
        return lamda_g
    lamda_g[c>cth] = (c[c>cth]/cth)**(2/3)
    lamda_g[c>b_l**-3] = cthb3**(-2/3)
    return lamda_g

# Deprecated
def correctForConcentrated(c, c88, lamda_g):
    lamda = 1 / lamda_g
    lamda[c>c88] = c[c>c88] / c88
    return lamda

# Deprecated
def getInitialGuesses(c, N, visc):
    """Plots normalized viscosity as well as initial guess based on the
    minimum values of the plots. If the minimum value corresponds to the
    actual plateau (it constitutes a reasonable guess), then the user should
    select 'Yes' in the pop-up dialog box. Otherwise, if this plateau should 
    not exist for this system (e.g., Bpe does not exist for neutral systems)
    then the user should select 'Ignore', and the guess will be sufficiently
    large as to be inconsequential to the fit. Finally, if the user feels that
    the guess should be elsewhere, they should select 'Let me pick', and
    select a plateau via the interactive tool.
    """

    plt.ion()
    fig, (ax0, ax1) = plt.subplots(1, 2)
    fig.set_size_inches(12, 6)
    t0, = ax0.plot([1], [1], 'ko')
    t1, = ax1.plot([1], [1], 'ko')
    ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    names = ['Bpe', 'Bg', 'Bth']
    nu_values = [1, 0.588, 0.5]
    p0 = np.zeros(4)
    for i, (name, nu) in enumerate(zip(names, nu_values)):
        fig.suptitle(name)
    
        plateau = visc / N / c**(1/(3*nu-1))
        t0.set_data(c, plateau)
        ax0.relim()
        ax0.autoscale(enable=True)
        
        plateau3 = visc / N / c**(3/(3*nu-1))
        t1.set_data(c, plateau3)
        ax1.relim()
        ax1.autoscale(enable=True)

        use_plat = mb.askyesno(
            title=f'{name} Plateau', 
            message='Does this regime exist?'
        )
        if not use_plat:
            p0[i] = 1e3
        else:
            p0[i] = min(plateau)**(1/3-nu)

    p0[3] = 5
    plt.close()
    plt.ioff()
    return p0

# To be implemented under a 'Save' function
def getSystemDescriptors(root, filepath):
    """Returns pandas dataframe containing specific viscosity data for a
    particular polymer/solvent pair. System identifiers should be requested
    and recorded in a file (e.g., polymer, solvent, temperature, salt).
    """
    rootname, _ = re.match('(.*)\.([A-Za-z0-9]+)$', filepath).groups()
    pickle_file = rootname + '.pickle'

    overwrite = True
    if os.path.exists(pickle_file):
        overwrite = mb.askyesno(
            title='File Exists', 
            message='An associated file already exists.'
            ' Do you want to overwrite it?'
        )
    if overwrite:
        new_w = tk.Toplevel(root)
        new_w.focus_force()
        infoApp = InfoDialog(new_w)

        system = infoApp.output
        #system_params = dict(zip(infoApp.entryNames, infoApp.outputs))

        try:
            new_w.destroy()
        except tk.TclError:
            pass
                
        with open(pickle_file, 'w') as f:
            print(asdict(system))
            pickle.dump(asdict(system), f)
    else:
        with open(pickle_file, 'r') as f:
            system = pickle.load(f)
    return system

# Deprecated
def getCrossoverConcentrations(Bpe, Bg, Bth, csfz=0):
    if csfz:
        cD = brentq(
            lambda c: Bg**2 - c**0.412 * (Bpe * np.sqrt(1 + 2*csfz/c))**0.764,
            0, 100
        )
    else:
        cD = Bpe**3 * (Bg / Bpe)**(1/0.206)
    cth = Bth**3 * (Bth / Bg)**(1/0.176)
    c88 = Bth**4
    b_l = Bth**-2
    return cD, cth, c88, b_l

# Deprecated, this is all in AnalysisApp now
def runAnalysis(root, filepath):

    df = pd.read_csv(filepath)

    analysis_w = tk.Toplevel(root)
    analysis_w.focus_force()
    analysisApp = AnalysisApp(analysis_w, df)
    '''
    # Clean data to semidilute only
    df = df[df.visc >= 0.95]

    c = df['c']
    N = df['N']
    visc = df['visc']

    p0 = getInitialGuesses(c, N, visc)
    
    # First run
    res = least_squares(
        lambda p: fitFunction(p, c, N) - visc,
        p0,
        method='trf',
        jac='3-point',
        bounds=((0, 0, 0, 2), 
                (1e10, 1e10, 1e10, 100)
        ),
        xtol=1e-10,
        verbose=0,
        loss='cauchy'
    )
    print(res.x)
    Bpe, Bg, Bth, Pe = res.x
    cD, cth, c88, b_l = getCrossoverConcentrations(Bpe, Bg, Bth) # c88 is c**
    print(cth*b_l**3)

    # TODO: If pe regime exists, check for residual salt in c < cD

    # If Rubinstein-Colby parameter indicates marginally good solvent, 
    # or if c > c** exists, then recalculate with lambda
    lamda_g = correctForRC(c, cth, b_l)
    lamda = correctForConcentrated(c, c88, lamda_g)

    g, regimes = DPinCorrBlob(Bpe, Bg, Bth, c, return_regimes=True)
    g *= lamda_g

    # Then refit Pe only
    res = least_squares(
        lambda pe: viscFunc1(N, pe, g) - lamda*visc,
        Pe,
        method='trf',
        jac='3-point',
        bounds=(0, 100),
        xtol=1e-10,
        verbose=0,
        loss='cauchy'
    )
    Pe, = res.x
    
    plt.plot(N/g/lamda_g, lamda*visc, 'ko')
    x_lin = np.linspace(1, max(N/g), 100)
    plt.plot(x_lin, viscFunc1(x_lin, Pe, 1), 'k-')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    '''

def main():
    root = tk.Tk()
    app = AnalysisApp(root)

    root.mainloop()

if __name__ == '__main__':
    main()