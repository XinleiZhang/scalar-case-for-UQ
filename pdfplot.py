#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 18:11:11 2018
https://python-graph-gallery.com/85-density-plot-with-matplotlib/
@author: zxl
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
#import seaborn as sns
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

ensemblemethod = 'EnRML' #'EnKF_MDA' 'EnKF'
casefolder = 'E2_'+ensemblemethod+'_iter30'
res = 'posterior' #'prior'# 'posterior'

# load file
x= np.loadtxt('./postprocessing/' + casefolder +'/' +res+'_x.txt')
y = np.loadtxt('./postprocessing/' + casefolder +'/' +res+'_y.txt')

plt.plot(x,y)

fig, ax = plt.subplots()

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_xlim([-0.4, 0.4])
ax.set_ylim([0,2])

ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2) #set_visible(False)
ax.spines['top'].set_linewidth(2) #set_visible(False)

colors = [(1,1,1)] + [(plt.cm.jet(i)) for i in range(1,256)]
mymap = mpl.colors.LinearSegmentedColormap.from_list('mymap', colors, N=256)

data = np.array([x, y])

kde = gaussian_kde(data)
nbins = 100
xx, yy = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j] #-0.4:0.4:0.001, 0:2:0.002]
density = kde(np.c_[xx.flat, yy.flat].T).reshape(xx.shape)
 
sns.kdeplot(x,y,n_levels=500,cmap=mymap, shade=True,shade_lowest=True,cbar=True)
ax = sns.kdeplot(x,y,n_levels=30,cmap=mymap, shade=True,shade_lowest=False,
            cbar=True)
cset = ax.contourf(xx, yy, density, 500, cmap=mymap, vmin=0,vmax=80)

cset = ax.pcolormesh(xx, yy, density, cmap=mymap, vmin=0, vmax=160,
                 shading='gouraud')
cbar = fig.colorbar(cset)
cbar.set_clim(0, 160)

plt.tight_layout()
plt.show()
fig.savefig('./figure/'+casefolder+'_'+res+'.png')
