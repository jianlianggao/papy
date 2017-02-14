## function of calcConfMatrixUniv
import sys
import numpy as np
from math import fabs

def calcConfMatrixUniv(p, corrVector, signThreshold, corrThresh):
   
    TP=0.0
    TN=0.0
    FP=0.0
    FN=0.0
    if (type(p).__name__ != 'ndarray'):
        p = np.array(p)
    Pf = p < signThreshold  
    try:
        nVars=p.shape[1]
    except IndexError:
        nVars=p.shape[0] 
    
    for i in range(0,nVars):
        if ((fabs(corrVector[i]) < corrThresh) and (Pf[0][i]==False)):
            TN=TN+1
        elif ((fabs(corrVector[i]) > corrThresh) and (Pf[0][i]==True)):
            TP=TP+1
        elif ((fabs(corrVector[i]) > corrThresh) and (Pf[0][i]==False)):
            FN=FN+1
        elif ((fabs(corrVector[i]) < corrThresh) and (Pf[0][i]==True)):
            FP=FP+1
        
    try:
        TNtot = TN/(FP+TN)
    except ZeroDivisionError:
        TNtot = float('NaN')
    try:
        TPtot = TP/(TP+FN)
    except ZeroDivisionError:
        TPtot = float('NaN')
    try:       
        FPtot = FP/(FP+TN)
    except ZeroDivisionError:
        FPtot = float('NaN')    
    try:
        FNtot = FN/(TP+FN)
    except ZeroDivisionError:
        FNtot = float('NaN')
    try:   
        FDtot = FP/(TP+FP)
    except ZeroDivisionError:
        FDtot = float('NaN')
        
    ## return the results
    try:
        return TNtot, TPtot, FPtot, FNtot, FDtot
    except:
        print "Errors occur when returning uncTNTot, uncTPTot, uncFPTot, uncFNTot, uncFDTot"

if __name__=="__main__":    
    calcConfMatrixUniv(p, corrVector, signThreshold, corrThresh)