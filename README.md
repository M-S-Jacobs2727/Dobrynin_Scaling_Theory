# Dobrynin_Scaling_Theory
Code that applies the polymer solution scaling theory outlined in the works of 
Dobrynin, Jacobs, and Sayko [1-5] to polymer solution viscosity data over a wide 
concentration range.

The ultimate goal of this project is for the user to load a web app, upload their data, 
and receive accurate values of scaling parameters $B_g$ and $B_{th}$ and entanglement 
packing number $P_e$, as well as moleucular characteristics such as Kuhn length, 
excluded volume, and thermal blob size.

## Folders

### *theoretical_nn_training*
This will be the production code cited in the eventual publication. I'm making it as
modular as possible, with a single config file to change all the settings. The user can
select a convolutional NN instead of a fully connected linear neural network by setting
the number of channels, kernel sizes, and pool sizes per convolution. The user can 
choose to build a model around a 2D or 3D representation of the generated surfaces by
setting the resolution to have length 2 or 3, respectively.

Additionally, the user can use `generators.py` (req. `data_processing.py`) with their
own ML code.

### *Molecular-Fingerprint-Code*
Ryan's individual setups for training CNN models on the generated data. Notably, he is 
using parallel computing to train his models much quicker on a higher resolution, 
currently 512x512.

### *CSV_files*
This contains a few .csv files of real, experimental data with three columns: 
concentration $\varphi$, $N_w$, and $\eta_{sp}$.

###  *Data* 
This contains theoretically generated data as well as experimental data studied in refs 
[1-5].

### *Mike gui test*
Mike's first attempt at creating a standalone gui for piecewise fitting of data to be 
loaded by the user. Will likely be scrapped for a web version.

## Glossary
| Term      | Explanation | Variable reference |
| ---------------- | ---------------- | ------------- |
| $c$ | polymer repeat unit concentration (number per unit volume)|
| $l$ | repeat unit projection length (e.g., nm) |
| $\varphi=cl^3$ | reduced polymer concentration | `phi` |
| $N_w$ | weight-average degree of polymerization (number of repeat units per chain) | `Nw` |
| $\eta_{sp}$ | specific viscosity | `eta_sp` |
| $B_g$ | good solvent scaling parameter | `Bg` |
| $B_{th}$ | thermal blob scaling parameter | `Bth` |
| $P_e$ | entanglement packing number[6-8] | `Pe` |


## Refs
1. Dobrynin, A. V.; Jacobs, M., When Do Polyelectrolytes Entangle? *Macromolecules* 2021, 54, 1859−1869.
2. Dobrynin, A. V.;  Jacobs, M.; Sayko, R., Scaling of Polymer Solutions as a Quantitative Tool. *Macromolecules* 2021, 54, 2288−2295.
3. Jacobs, M.;  Lopez, C. G.; Dobrynin, A. V., Quantifying the Effect of Multivalent Ions in Polyelectrolyte Solutions. *Macromolecules* 2021, 54, 9577−9586.
4. Sayko, R.; Jacobs, M.; Dobrynin, A. V., Quantifying Properties of Polysaccharide Solutions. *ACS Polymers Au* 2021, 1, 196–205.
5. Sayko, R.; Jacobs, M.; Dobrynin, A. V., Universality in Solution Properties of Polymers in Ionic Liquids. *ACS Appl. Polym. Mater.* 2022, 4, 1966–1973.
6. Kavassalis, T. A.; Noolandi, J., New View of Entanglements in Dense Polymer Systems. *Phys. Rev. Lett.* 1987, 59, 2674. 
7. Kavassalis, T. A.; Noolandi, J., A new theory of entanglements and dynamics in dense polymer systems. *Macromolecules* 1988, 21, 9, 2869–2879.
8. Kavassalis, T. A.; Noolandi, J., Entanglement scaling in polymer melts and solutions. *Macromolecules* 1989, 22, 6, 2709–2720.
