#!/usr/bin/python
#bionetworks.py
#last update : 14 Aug 2014

__author__ = '''Hyunju Kim'''


import networkx as nx
import os
import sys
import random as ran
from math import log
from optparse import OptionParser, OptionGroup
from scipy import *
from collections import defaultdict
import matplotlib.pyplot as plt
#from info_measure import *
import itertools
from pylab import *
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker
import operator
import copy

def a_line(xlist, ylist, axis_labels, file_name):
    plt.figure(figsize=(12,8))
    plt.plot(xlist, ylist, '-o')
    plt.xticks(xlist, axis_labels, rotation='vertical')
    plt.grid()
    plt.margins(0.2)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(file_name)
    plt.show()

def plot_AI_scale(dictO, result_file_name, viz_file_name):
    dictA = copy.deepcopy(dictO)
        #print "DDDDDDD"
    
    xlist = [x for x in range(len(dictA.keys()))]
    #list_node_names = list(dicA.keys())
    dictA_values = list(dictA.values())
    sorted_dicA_values = sorted(dictA_values)
    sorted_dicA_values.reverse()
    ylist = sorted_dicA_values
    axis_labels = []
    for u in ylist:
        for i, j in dictA.iteritems():
            if j == u:
                axis_labels.append(i)
                del dictA[i]
                break

    result_file = open(result_file_name, 'w')
    for i in range(len(xlist)):
        result_file.write('%s\t%f\n'%(axis_labels[i], ylist[i]))

    plt.figure(figsize=(12,8))
    plt.plot(xlist, ylist, '-o')
    plt.xticks(xlist, axis_labels, rotation='vertical')
    plt.grid()
    plt.margins(0.2)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(viz_file_name)
    #plt.show()


def plot_TE_scale(dictO, result_file_name, viz_file_name):
    dictA = copy.deepcopy(dictO)
    #print "DDDDDDD"
    
    xlist = [x for x in range(len(dictA.keys()))]
    #list_node_names = list(dicA.keys())
    dictA_values = list(dictA.values())
    sorted_dicA_values = sorted(dictA_values)
    sorted_dicA_values.reverse()
    ylist = sorted_dicA_values
    axis_labels = []
    for u in ylist:
        for i, j in dictA.iteritems():
            if j == u:
                axis_labels.append(i)
                del dictA[i]
                break

    result_file = open(result_file_name, 'w')
    for i in range(len(xlist)):
        result_file.write('%s\t%f\n'%(axis_labels[i], ylist[i]))

    plt.figure(figsize=(12,8))
    plt.plot(xlist, ylist, '-o')
    #plt.xticks(xlist, axis_labels, rotation='vertical')
    plt.grid()
    plt.margins(0.2)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(viz_file_name)
#plt.show()



def a_line_nolabel(xlist, ylist,  file_name):
    plt.figure(figsize=(12,8))
    plt.plot(xlist, ylist, '-o')
    plt.grid()
    plt.margins(0.2)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(file_name)
    #plt.show()


def heatmap(nodes_list, hpcell, output_file_name):
    
    xlabels = list(nodes_list)
#    hpcell = {}
#    print input_file_name
#    input_file = open(input_file_name, 'r')
#    for line in input_file:
#        items = [x.strip() for x in line.rstrip().split('\t')]
#        ynode = items[0]
#        xnode = items[1]
#        hpcell[(ynode, xnode)] = float(items[2])
    M = []
    for ynode in xlabels:
        rM = []
        for xnode in xlabels:
            rM.append(hpcell[(ynode, xnode)])
        M.append(rM)
    M = np.array(M)
    
    fig1 = plt.figure(figsize=(10,8))
    
    plt.yticks(np.arange(len(xlabels))+0.5, xlabels, size = 15, rotation=0, va="center", ha="right")
    plt.xticks(np.arange(len(xlabels))+0.5, xlabels, size = 15, rotation=90, va="top", ha="center")
    
    ax1 = fig1.add_subplot(111)
    cax1=ax1.pcolor(M, cmap=plt.cm.OrRd)
    plt.gca().set_aspect('equal')
    #ax1.set_title('Jaccard index of 27 pathways (edges)', size = 30)
    fig1.colorbar(cax1)
    plt.subplots_adjust(bottom=0.3)
    
    plt.savefig(output_file_name)
    #plt.show()


