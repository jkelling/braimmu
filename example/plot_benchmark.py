###########
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.ticker as ticker
from scipy.optimize import fsolve, curve_fit
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import axes3d
from math import sqrt, floor, pi
from matplotlib import rc, rcParams, cm
import matplotlib.mlab as mlab
from matplotlib.mlab import griddata
import subprocess

import numpy.ma as ma
from tempfile import mkstemp
from shutil import move
from os import fdopen, remove

import pandas as pd

import sys

rc('font',**{'family':'sans-serif','serif':['Times'], 'weight': 'bold', 'size' : 20})
# rc('text', usetex=True)
rcParams.update({'figure.autolayout': True})

def replace(file_path, line_tit, subst):
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                ls = line.split()
                if (len(ls) > 0) :
                    if (ls[0] == 'partition'):
                        line = subst
                new_file.write(line)
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)

def main():

    fig = pl.figure(figsize=(16,8))
    fig.set_tight_layout(False)
    
#    mly   = MultipleLocator(0.5)

    # read data frame 1
    df = pd.read_csv('benchmark.out', sep=' ', comment='#')

    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], facecolor=(1.,1.,1.))
    
    data_c = df[df.method == 'connectome']
#    data_cn = df[df.method == 'connectome']
#    data_n = df[df.method == 'connectome']
    
    coef = 1.0 / data_c.mu[0]
    
    ax.errorbar(data_c.ncore,data_c.mu * coef, yerr = data_c.sig / data_c.mu[0], markersize = 6, linestyle='', marker='s', color='r',label=r'$\rm vec$')
#    ax.errorbar(data_cn.ncore,data_cn.mu * coef, yerr = data_cn.sig * coef, markersize = 6, linestyle='', marker='^', color='g',label=r'$\rm vec/Newt$')
#    ax.errorbar(data_n.ncore,data_n.mu * coef, yerr = data_n.sig * coef, markersize = 6, linestyle='', marker='o', color='b',label=r'$\rm orig$')

    ax.set_xscale('log')    
    ax.set_yscale('log')    
    ax.set_xlabel(r'#cores',fontsize=20)
    ax.set_ylabel(r'Speedup',fontsize=20)
    ax.set_title(r'Without load balancing',fontsize=20)
    ax.set_ylim(0.9,12)
#    ax.yaxis.set_minor_locator(mly)
    ax.set_xticks([1, 2, 4, 6, 8, 10, 12])
    ax.set_yticks(range(1, 13, 2))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))


    # read data frame 2
#     df = pd.read_csv('benchmark_balance.out', sep=' ', comment='#')
# 
#     ax = fig.add_axes([0.55, 0.2, 0.4, 0.7], facecolor=(1.,1.,1.))
#     
#     data_c = df[df.method == 'connectome']
#     data_cn = df[df.method == 'connectome']
#     data_n = df[df.method == 'connectome']
#     
#     ax.errorbar(data_c.ncore,data_c.mu * coef, yerr = data_c.sig * coef, markersize = 6, linestyle='', marker='s', color='r',label=r'$\rm vec$')
#     ax.errorbar(data_cn.ncore,data_cn.mu * coef, yerr = data_cn.sig * coef, markersize = 6, linestyle='', marker='^', color='g',label=r'$\rm vec/Newt$')
#     ax.errorbar(data_n.ncore,data_n.mu * coef, yerr = data_n.sig * coef, markersize = 6, linestyle='', marker='o', color='b',label=r'$\rm orig$')
#    ax.legend(loc='lower right', fontsize=16, ncol=1)
#  
#    ax.set_xscale('log')    
#    ax.set_xlabel(r'$N_{\rm cores}$',fontsize=20)
#    ax.set_ylim(0.9,5.5)
#    ax.set_yticks([])
#    ax.yaxis.set_minor_locator(mly)
#    ax.set_title(r'${\rm With ~load ~balancing}$',fontsize=20)
    
    x = np.linspace(1, 12, 1000)
    ax.plot(x, x);
    pl.savefig("benchmark.png", format = 'png', dpi=200, orientation='landscape')
    #pl.show()
    #sys.stdin.read(1)
    pl.close()
    
if __name__ == '__main__':
    main()
