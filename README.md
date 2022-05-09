# Dobrynin_Scaling_Theory
Code that applies the polymer solution scaling theory outlined in the works of 
Dobrynin, Jacobs, and Sayko [1-5] to polymer solution viscosity data over a wide concentration range.

The ultimate goal of this project is for the user to load a web app, upload their data, and receive accurate
values of scaling parameters $B_g$ and $B_{th}$ and entanglement packing number $P_e$, as well as moleucular 
characteristics such as Kuhn length, excluded volume, and thermal blob size.

## Folder *CSV_files*
This contains a few .csv files of real, experimental data with three columns: concentration $\varphi$, $N_w$, and $\eta_{sp}$.

## Folder *Data* 
This contains theoretically generated data as well as all experimental data studied in refs [1-5].

## Folder *Mike gui test*
Mike's first attempt at creating a standalone gui for piecewise fitting of data to be loaded by the user. 
Will likely be scrapped for a web version?

## Folder *Mike_network_test*
Mike's attempt at generating theoretical data as needed while training a neural net.

## Folder *Molecular-Fingerprint-Code*
Ryan's libraries for building and running over 1.6 million ($\varphi$, $N_w$, $\eta_{sp}$) surfaces to load and train on.

## Glossary
| Term      | Explanation |
| ---------------- | ---------------- |
| $\varphi=cl^3$ | reduced polymer concentration |
| $c$ | polymer repeat unit concentration (units of mass/volume, number/volume, etc.)|
| $l$ | repeat unit projection length (units of nm, Angstrom, etc.) |
| $N_w$ | degree of polymerization (number of repeat units per chain) |
| $\eta_{sp}$ | specific viscosity |
| $B_g$ | good solvent scaling parameter |
| $B_{th}$ | thermal blob scaling parameter |
| $P_e$ | entanglement packing number[6-8] |


## Refs
1. Dobrynin, A. V.; Jacobs, M., When Do Polyelectrolytes Entangle? *Macromolecules* 2021, 54, 1859−1869.
2. Dobrynin, A. V.;  Jacobs, M.; Sayko, R., Scaling of Polymer Solutions as a Quantitative Tool. *Macromolecules* 2021, 54, 2288−2295.
3. Jacobs, M.;  Lopez, C. G.; Dobrynin, A. V., Quantifying the Effect of Multivalent Ions in Polyelectrolyte Solutions. *Macromolecules* 2021, 54, 9577−9586.
4. Sayko, R.; Jacobs, M.; Dobrynin, A. V., Quantifying Properties of Polysaccharide Solutions. *ACS Polymers Au* 2021, 1, 196–205.
5. Sayko, R.; Jacobs, M.; Dobrynin, A. V., Universality in Solution Properties of Polymers in Ionic Liquids. *ACS Appl. Polym. Mater.* 2022, 4, 1966–1973.
6. Kavassalis, T. A.; Noolandi, J., New View of Entanglements in Dense Polymer Systems. *Phys. Rev. Lett.* 1987, 59, 2674. 
7. Kavassalis, T. A.; Noolandi, J., A new theory of entanglements and dynamics in dense polymer systems. *Macromolecules* 1988, 21, 9, 2869–2879.
8. Kavassalis, T. A.; Noolandi, J., Entanglement scaling in polymer melts and solutions. *Macromolecules* 1989, 22, 6, 2709–2720.
