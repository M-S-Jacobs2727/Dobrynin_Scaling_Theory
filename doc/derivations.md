# Derivation of Specific Viscosity Equations

## General Case

In the following equations, unless otherwise noted, we assume $\nu=0.588$ for the good solvent exponent. The $\theta$-solvent/thermal blob exponent of $0.5$ is substituted for an explicit value.

We'll start with a few equations from the original works:

$$ \begin{align}
g &= \begin{cases}
    B_g^{3/(3\nu-1)} (cl^3)^{-1/(3\nu-1)} & \text{for } c \leq c_{th}\\
    B_{th}^{6} (cl^3)^{-2} & \text{for } c > c_{th}
\end{cases} \\ \nonumber \\
N_e &= P_e^2 \begin{cases}
    g(c_{th}b^3)^{2/3} &\text{for } & \quad c &\leq c_{th} \\
    B_{th}^2(cl^3)^{-4/3} &\text{for } c_{th} &< c &\leq b^{-3} \\
    g &\text{for } b^{-3} &< c &\leq c^{**} \\
    B_{th}^{-2}(c^{**}/c)^2 &\text{for } c^{**} &< c&
    \end{cases}  \\ \nonumber \\
\eta_{sp} &= N_w \left(1 + \left(\frac{N_w}{N_e}\right)^2\right) \begin{cases}
    g^{-1} &\text{for } c \leq c^{**} \\
    cbl^2 &\text{for } c > c^{**}
\end{cases}
\end{align} $$
If $c_{th} b^3 > 1$, then clearly the second condition for $N_e$ does not exist, and in fact the first condition also drops, leaving
$$ \begin{equation}
N_e = P_e^2 \begin{cases}
    g &\text{for } c \leq c^{**} \\
    B_{th}^{-2}(c^{**}/c)^2 &\text{for } c^{**} < c
\end{cases}
\end{equation} $$

To simplify this into a form suitable for efficient code, we would like to express $\eta_{sp}$ in the simplest terms possible. This can be achieved with the following definitions:

$$\begin{align}
c_{th}l^3 &= B_{th}^3 \left(\frac{B_{th}}{B_g}\right)^{1/(2\nu-1)} \\
c^{**}l^3 &= B_{th}^4 \\
b/l &= B_{th}^{-2} \\
\varphi &= cl^3
\end{align}$$
Additionally, we will use $\min$ and $\max$ functions rather than the piecewise functions to describe the transition from one state to another across concentration regimes. For example, equation 1 can be rewritten as
$$\begin{equation}
    g= \min_c\left[B_g^{3/(3\nu-1)} \varphi^{-1/(3\nu-1)}, \; B_{th}^{6} \varphi^{-2} \right]
\end{equation}$$

For $N_e$, we will first substitute for the unwanted parameters (equations 5-7):
$$\begin{equation}
    N_e = P_e^2 \begin{cases}
        B_g^{3/(3\nu-1)} \varphi^{-1/(3\nu-1)} \left(B_{th}^3 \left(\frac{B_{th}}{B_g}\right)^{1/(2\nu-1)} B_{th}^{-6}\right)^{2/3} &\text{for } &\quad c &\leq c_{th} \\
        B_{th}^2\varphi^{-4/3} &\text{for } c_{th} &< c &\leq b^{-3} \\
        B_{th}^{6} \varphi^{-2} &\text{for } b^{-3} &< c &\leq c^{**} \\
        B_{th}^{6} \varphi^{-2} &\text{for } c^{**} &< c&
    \end{cases}
\end{equation}$$
A quick aside: the expressions for the last two cases are identical for $N_e$, but are kept separate in the articles to preserve the understanding of the two different mechanisms that generate these concentration dependences, and to contrast it with the equation for specific viscosity (equation 3) which does change across that boundary.

We can simplify the first case to 
$$\begin{align}
B_g^{(\frac{3}{3\nu-1} - \frac{2}{3}\frac{1}{2\nu-1})}
B_{th}^{\frac{2}{3}(-3+\frac{1}{2\nu-1})}
\varphi^{-1/(3\nu-1)}
&= B_g^{\frac{12\nu-7}{(6\nu-3)(3\nu-1)}}
B_{th}^{-\frac{4}{3}(\frac{3\nu-2}{2\nu-1})}
\varphi^{-1/(3\nu-1)} \nonumber \\
&= B_g^{\frac{0.056}{(0.528)(0.764)}} B_{th}^{\frac{0.944}{0.528}} \varphi^{\frac{-1}{0.764}}
\end{align} $$
where in the last step we substituted $\nu=0.588$. As these are exponents, this last expression is as simplified as we'd like to make it, since any round-off error ahead of time would have significant impacts down the line. 
Noting that $1/0.764\approx 1.31$ and $4/3 \approx 1.33$, the transition from the first case to the second uses another $\min$ function, as does the transition from the second to the third. Therefore, we can write equation 2 as
$$\begin{align}
N_e = P_e^2 \times \min_c \Bigg[
    B_g^{\frac{0.056}{(0.528)(0.764)}} B_{th}^{\frac{0.944}{0.528}} \varphi^{\frac{-1}{0.764}},
    \min_c \bigg[ B_{th}^2\varphi^{-4/3}, \; B_{th}^{6} \varphi^{-2} \bigg]
\Bigg]
\end{align}$$

Finally, we could express $\eta_{sp}$ in simplest terms by substituting in equations 9 and 12, but the resulting code would be nigh unparseable. So, we settle for computing equations 9 and 12 as intermediate steps, and substitute the resulting tensors into 
$$\begin{equation}
    \eta_{sp} = N_w \left(1 + \left(\frac{N_w}{N_e}\right)\right)^2 
    \times \min_c \Big[g^{-1}, \; \varphi b/l \Big]
\end{equation}$$
The minimum is valid here in all cases, as $g^{-1} \propto \varphi^{\frac{1}{3\nu-1}\approx 1.31}$ in the good solvent regime, $g^{-1} \propto \varphi^{2}$ in the thermal blob regime, so the first term in the bracketed expression scales more strongly than the second. 

For added clarity, equations 1-3 are shown in the figures below (on a log-log scale), with dashed lines indicating where a respective case of a piecewise function would follow if it extended beyond its dedicated concentration regime, which are separated by vertical lines. The horizontal lines show important values at the crossover points between regimes. 

## Case of Athermal Solvent ($c_{th} \to \infty$)

In the case that the thermal blobs shrink to or below the size of the Kuhn length, we consider the solvent to be athermal for the polymer chains, and the thermal blob overlap concentration diverges. Thus, we have the following three equations:

$$\begin{align}
    g &= (B_g^3/ \varphi)^{1/0.764} \\
    N_e &= P_e^2 g \\
    \eta_{sp} &= N_w \left(1 + \left(\frac{N_w}{N_e}\right)^2\right)
    \begin{cases}
        g^{-1} &\text{for } c \leq c^{**} \\
        \varphi b/l &\text{for } c > c^{**}
    \end{cases}
\end{align}$$
To compute the Kuhn length in an athermal solvent, we observe that the correlation length in the general case at concentrations $c^* \leq c \leq c_{th}$ can be written as
$$\begin{equation}
    \xi = \left( \frac{g}{g_{th}} \right)^\nu D_{th}
\end{equation}$$
where $D_{th}$ is the thermal blob size, $g_{th}$ is the number of repeat units per thermal blob, and $g/g_{th}$ is the number of thermal blobs per correlation blob. In the case of an athermal solvent, we can show that the correlation length can instead be written in terms of the Kuhn length $b$ as
$$\begin{equation}
    \xi = \left( \frac{gl}{b} \right)^\nu b
\end{equation}$$
where $gl/b$ is the number of Kuhn segments within the correlation blob. Using the space-filling condition $g/\xi^3 = c$ and the previous definition for $g$ in terms of $B_g$, we can show that
$$\begin{equation}
    B_g = (l/b)^{1-\nu}
\end{equation}$$
or
$$\begin{equation}
    b = l B_g^\frac{1}{0.412}
\end{equation}$$

## Case of $\Theta$ Solvent ($c_{th} < c^*$)

In this case, the thermal blobs expand to sizes larger than the chain in an ideal conformation and the initial equations are transformed as
$$\begin{align}
    g &= B_{th}^6/ \varphi^2 \\
    N_e &= P_e^2 \times \min_c \Big[ B_{th}^2\varphi^{-4/3}, \; B_{th}^{6} \varphi^{-2} \Big] \\
    \eta_{sp} &= N_w \left(1 + \left(\frac{N_w}{N_e}\right)^2\right) \times 
    \min_c \Big[
        g^{-1}, \; \varphi b/l
    \Big]
\end{align}$$
