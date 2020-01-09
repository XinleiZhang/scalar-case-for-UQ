#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib as mpl

# set plot properties
params = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'axes.grid': False,
    'savefig.dpi': 150,
    'axes.labelsize': 30, # 10
    'axes.titlesize': 30,
    'font.size': 30,
    'legend.fontsize': 30,
    'xtick.labelsize': 35,
    'ytick.labelsize': 35,
    'text.usetex': False,
    'figure.figsize': [8, 6],
    'font.family': 'serif',
}

mpl.rcParams.update(params)

# define which case to plot
ensemblemethod = 'EnKFMDA' #'EnRML' EnKF_MDA' 'EnKF'
res = 'posterior' #'prior'or 'posterior'
casefolder = 'E4_'+ensemblemethod+'_iter30'

# load file
x= np.loadtxt('./postprocessing/' + casefolder +'/' +res+'_x.txt')
y = np.loadtxt('./postprocessing/' + casefolder +'/' +res+'_y.txt')

fig, ax = plt.subplots()

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_xlim([-0.4, 0.4])
ax.set_ylim([0,2])

ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

colors = [(1,1,1)] + [(plt.cm.jet(i)) for i in range(1,256)]
mymap = mpl.colors.LinearSegmentedColormap.from_list('mymap', colors, N=256)

data = np.array([x, y])

kde = gaussian_kde(data)
nbins = 100
xx, yy = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
density = kde(np.c_[xx.flat, yy.flat].T).reshape(xx.shape)

cset = ax.pcolormesh(xx, yy, density, cmap=mymap, vmin=0, vmax=160,
                 shading='gouraud')
cbar = fig.colorbar(cset)
cbar.set_clim(0, 160)

plt.tight_layout()
plt.show()
fig.savefig('./figure/'+casefolder+'_'+res+'.png')
