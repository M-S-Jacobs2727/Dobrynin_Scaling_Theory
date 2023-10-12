import numpy as np
import matplotlib.pyplot as plt


def plot_g():
    x = np.linspace(0, 1, 101, endpoint=True)
    a = 1
    b = 0.6911/2+a

    xc = 0.5
    yc = b-2*xc
    
    good_line = a-1.3089*x
    th_line = b-2*x
    solid = np.minimum(good_line, th_line)
    dashed = np.maximum(good_line, th_line)
    
    plt.plot(x, solid, 'k-', lw=8)
    plt.plot(x, dashed, 'k--', lw=3)
    
    plt.plot([xc, xc], [solid.min(), dashed.max()], 'k-', lw=0.5)
    plt.plot([0, 1], [yc, yc], 'k-', lw=0.5)
    
    plt.xlim((0, 1))
    plt.ylim((solid.min(), dashed.max()))
    
    plt.xticks([])
    plt.yticks([])
    plt.xticks([xc,], labels=["$c_{th}$",])
    plt.yticks([yc,], labels=["$g_{th}$",])
    
    plt.savefig("img/figure_g.png")


def plot_ne():
    x = np.linspace(0, 1, 101, endpoint=True)

    xc = np.array([0.33, 0.67])
    yc = b-4/3*xc
    
    a = 1
    b = 0.6911/2+a

    lines = [a-1/0.764*x, b-4/3*x, 3*b-2*x]
    
    solid = np.minimum(lines[0], np.minimum(lines[1], lines[2]))
    dashed1 = np.maximum(lines[0], lines[1])
    dashed2 = np.maximum(lines[1], lines[2])
    
    plt.plot(x, solid, 'k-', lw=8)
    plt.plot(x, dashed1, 'k--', lw=3)
    plt.plot(x, dashed2, 'k--', lw=3)
    
    plt.plot([xc, xc], [lines[2].min(), lines[2].max()], 'k-', lw=0.5)
    plt.plot([0, 1], [yc, yc], 'k-', lw=0.5)

    plt.xlim((0, 1))
    plt.ylim((lines[2].min(), lines[2].max()))

    plt.show()


def main():
    plot_ne()

if __name__ == "__main__":
    main()
