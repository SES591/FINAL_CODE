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
import numpy as np
from collections import defaultdict




################# begin : build_historyList ############################
def build_historyList(aList, historyLength, Nbr_States=2):
    '''
    Description:
        -- To generate a list of decimal states for k-previous states of a node with a given list of dynamical states of the nodes
     
    Arguments:
        -- 1. aList = a given list of dynamical states of a node
        -- 2. historyLength = the number of previous states converted to a decimal state
        -- 3. Nbr_States = the number of possible states for each node (2 by default)
          
    Return:
        -- a list of decimal states for k-previous states of a node
    '''

    historyList = []
    historyUnit = aList[:historyLength]
    aList[:historyLength] = []
    
    historyState = 0
    for s in range(historyLength):
        historyState += historyUnit[s] * np.power(Nbr_States, s)
    historyList.append(historyState)

    for x in aList:
        historyState = historyState / Nbr_States + x * np.power(Nbr_States, historyLength - 1)
        historyList.append(historyState)

    return historyList
################# end : build_historyList ########################


################## begin: compute_AI ########################
def compute_AI(timeSeriesNode, historyLength, Nbr_Initial_States, Nbr_States=2):
    '''
    Description:
    -- compute AI for every node using distribution from all possible initial conditions or an arbitrary set of initial conditions
    
    Note:
    -- for an arbitrary set of initial conditions, one need to specify the number of initial conditions, Nbr_Initial_States, and name every initial condition from 0 to Nbr_Initial_States - 1
    '''
    count_currState_hiState = defaultdict(int)
    count_hiState = defaultdict(int)
    count_currState = defaultdict(int)
    
    for si in range(Nbr_Initial_States):
        aList = list(timeSeriesNode[si])
        historyList = build_historyList(aList, historyLength, Nbr_States) # aList becomes aList[historyLength:] after historyList function
        
        for s in range(len(aList)):
            count_currState_hiState[(aList[s], historyList[s])] += 1
            count_hiState[historyList[s]] += 1
            count_currState[aList[s]] += 1

    AI = 0
    for si in range(Nbr_Initial_States):
        aList = list(timeSeriesNode[si])
        historyList = build_historyList(aList, historyLength, Nbr_States) # aList becomes aList[historyLength:] after historyList function
        
        sampleLength = len(aList) * Nbr_Initial_States
        for s in range(len(aList)):
            prob_currState_hiState = float(count_currState_hiState[(aList[s], historyList[s])]) / float(sampleLength)
            prob_hiState = float(count_hiState[historyList[s]]) / float(sampleLength)
            prob_currState = float(count_currState[aList[s]]) / float(sampleLength)
            AI = AI + log( prob_currState_hiState / ( prob_currState * prob_hiState)) / log(2.0) # since the summation is over not all possible pattern of currState_hiState
    AI = AI / float(sampleLength)
    return AI
################## end : compute_AI ########################


################## begin : compute_TE ########################
def compute_TE(timeSeriesNodeA, timeSeriesNodeB, historyLength, Nbr_Initial_States, Nbr_States=2):
    '''
    Description:
    -- compute TE for every pair of nodes using distribution from all possible initial conditions or an arbitrary set of initial conditions
    
    Note:
    -- for an arbitrary set of initial conditions, one need to specify the number of initial conditions, Nbr_Initial_States, and name every initial condition from 0 to Nbr_Initial_States - 1
    '''
    #*** declare dic for distribution to compute Transfer Entropy ***#
    count_tarCurrState_tarHiState_sourPrevState = defaultdict(int)
    count_tarCurrState_tarHiState = defaultdict(int)
    count_tarHiState_sourPrevState = defaultdict(int)
    count_tarHiState = defaultdict(int)
    
    for si in range(Nbr_Initial_States):
        sourList = list(timeSeriesNodeA[si])
        tarList = list(timeSeriesNodeB[si])
        historyList = build_historyList(tarList, historyLength) # tarList becomes tarList[historyLength:] after historyList function
        sourList[:historyLength - 1] = [] # sourList becomes sourList[historyLength-1:]
        #*** To obtain the distribution for each pattern to compute Transfer Entropy ***#
        for s in range(len(tarList)):
            count_tarCurrState_tarHiState_sourPrevState[(tarList[s], historyList[s], sourList[s])] += 1
            count_tarCurrState_tarHiState[(tarList[s], historyList[s])] += 1
            count_tarHiState_sourPrevState[(historyList[s], sourList[s])] += 1
            count_tarHiState[historyList[s]] += 1

    #*** obtain the distribution for each pattern to compute Active Information ***#
    TE = 0
    for si in range(Nbr_Initial_States):
        sourList = list(timeSeriesNodeA[si])
        tarList = list(timeSeriesNodeB[si])
        historyList = build_historyList(tarList, historyLength) # tarList becomes tarList[historyLength:] after historyList function
        sourList[:historyLength - 1] = [] # sourList becomes sourList[historyLength-1:]
        sampleLength = len(tarList) * Nbr_Initial_States
        for s in range(len(tarList)):
            prob_tarCurrState_tarHiState_sourPrevState = float(count_tarCurrState_tarHiState_sourPrevState[(tarList[s], historyList[s], sourList[s])]) / float(sampleLength)
            prob_tarCurrState_tarHiState = float(count_tarCurrState_tarHiState[(tarList[s], historyList[s])]) / float(sampleLength)
            prob_tarHiState_sourPrevState = float(count_tarHiState_sourPrevState[(historyList[s], sourList[s])]) / float(sampleLength)
            prob_tarHiState = float(count_tarHiState[historyList[s]]) / float(sampleLength)
            TE = TE + log( (prob_tarCurrState_tarHiState_sourPrevState * prob_tarHiState) / ( prob_tarHiState_sourPrevState * prob_tarCurrState_tarHiState)) / log(2.0) # since the summation is over not all possible pattern of tarCurrState_tarHiState_sourPrevState but tarList, there is no prob_tarCurrState_tarHiState_sourPrevState multiplied by the log term.
    TE = TE / float(sampleLength)
    return TE
################## end : compute_TE ########################



timeSeriesNodeA = {0: [0,0,1, 0, 0, 1, 1, 1]}#, 1:[0,0,1, 0, 0, 0, 1, 0] }
timeSeriesNodeB = {0: [0,0,1, 0, 1, 0, 0, 1]}#, 1:[1,1,1, 0, 0, 0, 1, 1] }

timeSeriesNodeA = {0: [0,0,1, 0, 0, 1, 1, 1], 1:[0,0,1, 0, 0, 0, 1, 0] }

#print compute_local_TE(timeSeriesNodeA[0], timeSeriesNodeB[1], 2, 1,2)

#hList = build_historyList(aList, 2)
#print "history list in decimal states", hList

#print compute_AI(timeSeriesNode, 2, 2)

#tarList = [1,1,1,0,1,1,0,1,0,0,1,1,0,0,1,1,1,0,1,1,0,1]
#sourList = [1,0,1,0,1,0,1,0,1,1,0,1,1,0,1,1,0,1,1,0,0,1]
#print comTE(sourList, tarList, 1)

