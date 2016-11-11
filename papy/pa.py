#!/usr/bin/env python
"""
Power Analysis (calculation) tool
Developed by Dr. Goncalo Correia and Dr Jianliang Gao
Imperial College London
2016
"""
import os,sys,csv,inspect,dis,os.path,random,multiprocessing
import numpy as np
import scipy.stats as scistats
import statsmodels.formula.api as sm                    #for linear regression
import matplotlib.pyplot as plt
from math import fabs,floor,ceil,log,exp
from datetime import datetime
from joblib import Parallel, delayed                    #for Parallel computing
# For 3d plots. This import is necessary to have 3D plotting below
from mpl_toolkits.mplot3d import Axes3D
# for saving the plot to pdf file 
from matplotlib.backends.backend_pdf import PdfPages

##=======Beginning of SurfacePlot=========================
def SurfacePlot(output, variable,metric,correction, sizeeff,samplsizes,nreps):
    MUtot = output[variable-1][correction-1][metric-1]
    NS, NSE = MUtot.shape
    SIGMAtot = output[variable-1][correction-1][metric+5-1]
    SIGMAlow=MUtot-1.96*SIGMAtot/np.sqrt(nreps)
    SIGMAlow = np.array([[0 if x<0 else x for x in y] for y in SIGMAlow])
    
    ##plot
    #generate a 2D grid
    X, Y = np.meshgrid(sizeeff, samplsizes)
    
    # Plot the data
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, MUtot, cmap=plt.cm.coolwarm,rstride=1, cstride=1, alpha = 0.5)
    #create a contour of the surface on z axis
    cset = ax.contourf(X, Y, MUtot, zdir='z', offset=-0.5, cmap=plt.cm.coolwarm, alpha = 0.5)
    ax.view_init(20, -120)
    ax.set_xlabel('Sample size')
    ax.set_ylabel('Effect Size')
    ax.set_zlabel('Rate')
    #plt.gca().invert_xaxis()
    ax.set_zlim(-0.5,1.5)
    
    #for saving the plot to pdf file
    #To make a multi-page pdf file, first initialize the file:
    pp = PdfPages('multipage.pdf')
    
    #give the PdfPages object to savefig()
    plt.savefig(pp, format='pdf')
    pp.savefig()
    pp.close()
    
    #plt.show()
##=======End of SurfacePlot=========================

##=======Beginning of simulateLogNormal===================
def simulateLogNormal(data, covType, nSamples):
    ## find offset and offset the data. Why to do this?
    offset = fabs(np.amin(data)) + 1
    offData = data + offset
    
    
    ##log on the data array
    logData = np.log(offData)
    

    meansLog = np.mean(logData, axis=0)
    
    if (covType=='Estimate'):
        covLog=np.cov(logData, rowvar=0)
    elif (covType=='Diagonal'):
        varlogData=np.var(logData,axis=0)       #get variance of log data by each column
        covLog=np.diag(varlogData)               #generate a matrix with diagonal of variance of log Data
    else:
        print 'Unknown Covariance type'   
        
    simData = np.random.multivariate_normal(np.transpose(meansLog),covLog,nSamples)
    
    simData = np.exp(simData)
    simData = simData - offset
    
    ##Set to 0 negative values 
    simData = [[0 if x<0 else x for x in y] for y in simData]
    if (type(simData).__name__ != 'ndarray'):
        simData = np.array(simData)        
    corrMatrix = np.corrcoef(simData, rowvar=0)   #work out the correlation of matrix by columns, each column is a variable
    
    return simData, corrMatrix
##=======End of simulateLogNormal===================


##=======Beginning of PCalc_Continuous====================
def PCalc_Continuous(data, EffectSizes, SampSizes, SignThreshold, nSimSamp, nRepeat):
    ##If sample size bigger than number of simulated samples adjust it
    ## global sampSizes, signThreshold, effectSizes, nRepeats, nSampSizes, nEffSizes, Samples_seg, correlationMat_seg, output2
    sampSizes = SampSizes
    signThreshold = SignThreshold
    effectSizes = EffectSizes
    
    try:
        if max(sampSizes) >= nSimSamp:
            print 'Number of simulated samples smaller than maximum of samplesizes to check - increased'
            nSimSamp = max(sampSizes) + 500
    except ValueError:
        if max(max(sampSizes)) >= nSimSamp:
            print 'Number of simulated samples smaller than maximum of samplesizes to check - increased'
            nSimSamp = max(max(sampSizes)) + 500
    ## convert matrix to numpy array type if needed
    if (type(data).__name__ != 'ndarray'):
        data = np.array(data)
    
    ##get data array size
    size = data.shape
    rows = size[0]
    if (data.ndim > 1):
        cols = size[1]
    else:
        cols = 1    
        
    ##Number of variables
    numVars = cols
    nRepeats = nRepeat
    
    ##Number of sample and effect sizes
    if (sampSizes.ndim >1):
        nSampSizes = sampSizes.shape[1]
    elif (sampSizes.ndim ==1):
        nSampSizes = sampSizes.shape[0]
        
    if (effectSizes.ndim >1):
        nEffSizes = effectSizes.shape[1]
    elif (effectSizes.ndim ==1):
        nEffSizes = effectSizes.shape[0]
    

    ##Simulation of a new data set based on multivariate normal distribution
    Samples, correlationMat = simulateLogNormal(data,'Estimate', nSimSamp)
    
    ##split Samples and correlationMat into chunk files for parallel processing
    if multiprocessing.cpu_count()-1 <= 0:
        cores = 1
    else:
        cores = multiprocessing.cpu_count()
    Samples_seg = _chunkMatrix(Samples, cores)
    correlationMat_seg = _chunkMatrix(correlationMat, cores)
    ## Initialize the output structures
    ##output2 = np.zeros((cores,1, 4, nRepeats, nEffSizes,nSampSizes)) 
    output2=[]
    ## for ii in range(cores):
        ## output2.append([])  
    
    output2 = Parallel(n_jobs=cores)(delayed(f_multiproc1)(sampSizes, signThreshold, effectSizes, nRepeats, nSampSizes, nEffSizes, Samples_seg, correlationMat_seg,ii) for ii in range(cores))            #n_jobs=-1 means using all CPUs or any other number>1 to specify the number of CPUs to use
    ## for ii in range(numVars):       # non parallelized loop
        ## f_multiproc(ii)
    ##pass the results to output
    output = np.array([])
    output = np.array(output2[0])
    for ii in range(1, cores):
        #debug
        #print (np.array(output2[ii])).shape
        output = np.append(output, output2[ii],axis=0)
          
    try:
        return output
    except:
        print 'error occurs when returning output'

def f_multiproc1(sampSizes, signThreshold, effectSizes, nRepeats, nSampSizes, nEffSizes, Samples_seg, correlationMat_seg, currCore):
    
    ## global Samples_seg, correlationMat_seg, nEffSizes, nSampSizes, nRepeats, sampSizes, effectSizes, output2
    #re-check numVars
    numVars = Samples_seg[currCore].shape[1]
    
    output = []
    
    if (nEffSizes == 1 and nSampSizes == 1):
        storeVar = np.zeros((4,nRepeats))
    elif (nEffSizes > 1 or nSampSizes > 1):
        storeVar = np.zeros((4,nRepeats, nEffSizes, nSampSizes))
        
    #define uncStruct -- structual data
    uncStruct = {'TP':np.zeros((nEffSizes, nSampSizes)),'FP':np.zeros((nEffSizes, nSampSizes)),'TN':np.zeros((nEffSizes, nSampSizes)),\
                 'FN':np.zeros((nEffSizes, nSampSizes)),'FD':np.zeros((nEffSizes, nSampSizes)),'STP':np.zeros((nEffSizes, nSampSizes)),\
                 'SFP':np.zeros((nEffSizes, nSampSizes)),'STN':np.zeros((nEffSizes, nSampSizes)),'SFN':np.zeros((nEffSizes, nSampSizes)),\
                 'SFD':np.zeros((nEffSizes, nSampSizes))}
    
    bonfStruct = {'TP':np.zeros((nEffSizes, nSampSizes)),'FP':np.zeros((nEffSizes, nSampSizes)),'TN':np.zeros((nEffSizes, nSampSizes)),\
                 'FN':np.zeros((nEffSizes, nSampSizes)),'FD':np.zeros((nEffSizes, nSampSizes)),'STP':np.zeros((nEffSizes, nSampSizes)),\
                 'SFP':np.zeros((nEffSizes, nSampSizes)),'STN':np.zeros((nEffSizes, nSampSizes)),'SFN':np.zeros((nEffSizes, nSampSizes)),\
                 'SFD':np.zeros((nEffSizes, nSampSizes))}
    
    bhStruct = {'TP':np.zeros((nEffSizes, nSampSizes)),'FP':np.zeros((nEffSizes, nSampSizes)),'TN':np.zeros((nEffSizes, nSampSizes)),\
                 'FN':np.zeros((nEffSizes, nSampSizes)),'FD':np.zeros((nEffSizes, nSampSizes)),'STP':np.zeros((nEffSizes, nSampSizes)),\
                 'SFP':np.zeros((nEffSizes, nSampSizes)),'STN':np.zeros((nEffSizes, nSampSizes)),'SFN':np.zeros((nEffSizes, nSampSizes)),\
                 'SFD':np.zeros((nEffSizes, nSampSizes))}
    
    byStruct = {'TP':np.zeros((nEffSizes, nSampSizes)),'FP':np.zeros((nEffSizes, nSampSizes)),'TN':np.zeros((nEffSizes, nSampSizes)),\
                 'FN':np.zeros((nEffSizes, nSampSizes)),'FD':np.zeros((nEffSizes, nSampSizes)),'STP':np.zeros((nEffSizes, nSampSizes)),\
                 'SFP':np.zeros((nEffSizes, nSampSizes)),'STN':np.zeros((nEffSizes, nSampSizes)),'SFN':np.zeros((nEffSizes, nSampSizes)),\
                 'SFD':np.zeros((nEffSizes, nSampSizes))}
    
    for currVar in range(0,numVars):
        for currEff in range(0,nEffSizes):
            b1 = np.zeros((numVars,1))
            b1[currVar][0] = effectSizes[0][currEff]
            
            for currSampSize in range(0,nSampSizes):
                # define the structural data multiplerepeats
                class MUltiplerepeats(object):
                    def __init__(self,Results,Bonferroni,BenjHoch,BenjYek,noCorrection):
                        self.Results = Results
                        self.Bonferroni = Bonferroni
                        self.BenjHoch = BenjHoch
                        self.BenjYek = BenjYek
                        self.noCorrection = noCorrection
                multiplerepeats=MUltiplerepeats({'TP':np.zeros(nRepeats),'FP':np.zeros(nRepeats),'TN':np.zeros(nRepeats),'FN':np.zeros(nRepeats),'FD':np.zeros(nRepeats)},\
                                                {'TP':np.zeros(nRepeats),'FP':np.zeros(nRepeats),'TN':np.zeros(nRepeats),'FN':np.zeros(nRepeats),'FD':np.zeros(nRepeats)},\
                                                {'TP':np.zeros(nRepeats),'FP':np.zeros(nRepeats),'TN':np.zeros(nRepeats),'FN':np.zeros(nRepeats),'FD':np.zeros(nRepeats)},\
                                                {'TP':np.zeros(nRepeats),'FP':np.zeros(nRepeats),'TN':np.zeros(nRepeats),'FN':np.zeros(nRepeats),'FD':np.zeros(nRepeats)},\
                                                {'TP':np.zeros(nRepeats),'FP':np.zeros(nRepeats),'TN':np.zeros(nRepeats),'FN':np.zeros(nRepeats),'FD':np.zeros(nRepeats)})
                
                for currRepeat in range(0, nRepeats):
                    ## Select a subset of the simulated spectra
                    selectIndex = randperm1(sampSizes[0][currSampSize])
                    
                    if (type(selectIndex).__name__ != 'ndarray'):
                        selectIndex = np.array(selectIndex)
                    SelSamples = Samples_seg[currCore][selectIndex]                    # matrix slicing
                                    
                    # UVScaling the data - vectorize with bsxfun 
                    stDev = np.std(SelSamples, axis=0)   # without argument ddof=1 means using default ddof=0 to work out std on population
                    SelSamples = SelSamples - np.mean(SelSamples, axis=0)
                    SelSamples =  SelSamples/stDev
                    
                    noiseLevel = 1
                    noise = noiseLevel*np.random.randn(sampSizes[0][currSampSize],1)
                    
                    Y = SelSamples[:, np.array([currVar])]*b1[currVar][0]
                    Y = Y + noise
                                        
                    p = np.zeros((1,numVars))
                    
                    #Using regress for multivariate regression test
                    for i in range(0, numVars):
                        B = np.append(np.ones((Y.shape[0],1)), SelSamples[:,[i]], 1)
                        stats_result = sm.OLS(Y,B).fit()                    # ordinary least square linear regression
                                                                            # OLS. The result of OLS has attributes such as
                                                                            # .rsquared as R^2, .fvalue as F-statistics
                                                                            # .f_pvalue as p-value of F-stats, .scale as error variance
                        
                        p[0][i] = stats_result.f_pvalue
                    
                    pUnc = p                ##pUnc and p have 1xnumVars elements
                    pBonf = p * numVars
                    
                    h1, crit_p, adj_ci_cvrg, pBY = fdr_bh(p, 0.05, 'dep')
                    h1, crit_p, adj_ci_cvrg, pBH = fdr_bh(p, 0.05, 'pdep')
                    
                    #need to debug below
                    corrVector = correlationMat_seg[currCore][:,currVar]
                    
                    uncTNTot, uncTPTot, uncFPTot, uncFNTot, uncFDTot = calcConfMatrixUniv(pUnc, corrVector, signThreshold, 0.8)
                    bonfTNTot, bonfTPTot, bonfFPTot, bonfFNTot, bonfFDTot = calcConfMatrixUniv(pBonf, corrVector, signThreshold, 0.8)
                    
                    byTNTot, byTPTot, byFPTot, byFNTot, byFDTot = calcConfMatrixUniv(pBY, corrVector, signThreshold, 0.8)
                    bhTNTot, bhTPTot, bhFPTot, bhFNTot, bhFDTot = calcConfMatrixUniv(pBH, corrVector, signThreshold, 0.8)
                    
                    try:
                        multiplerepeats.noCorrection['TP'][currRepeat] = uncTPTot
                    except IndexError:
                        multiplerepeats.noCorrection['TP'] = np.append(multiplerepeats.noCorrection['TP'], uncTPTot)  #if array index exceeds upper bound, extend the array
                    try:    
                        multiplerepeats.noCorrection['FP'][currRepeat] = uncFPTot
                    except IndexError:
                        multiplerepeats.noCorrection['FP'] = np.append(multiplerepeats.noCorrection['FP'], uncFPTot)
                    try:
                        multiplerepeats.noCorrection['TN'][currRepeat] = uncTNTot
                    except IndexError:
                        multiplerepeats.noCorrection['TN'] = np.append(multiplerepeats.noCorrection['TN'], uncTNTot)
                    try:
                        multiplerepeats.noCorrection['FN'][currRepeat] = uncFNTot
                    except IndexError:
                        multiplerepeats.noCorrection['FN'] = np.append(multiplerepeats.noCorrection['FN'], uncFNTot)
                    try:
                        multiplerepeats.noCorrection['FD'][currRepeat] = uncFDTot
                    except IndexError:
                        multiplerepeats.noCorrection['FD'] = np.append(multiplerepeats.noCorrection['FD'], uncFDTot)
                        
                        
                    try:
                        multiplerepeats.Bonferroni['TP'][currRepeat] = bonfTPTot
                    except IndexError:
                        multiplerepeats.Bonferroni['TP'] = np.append(multiplerepeats.Bonferroni['TP'], bonfTPTot)
                    try:
                        multiplerepeats.Bonferroni['FP'][currRepeat] = bonfFPTot
                    except IndexError:
                        multiplerepeats.Bonferroni['FP'] = np.append(multiplerepeats.Bonferroni['FP'], bonfFPTot)
                    try:
                        multiplerepeats.Bonferroni['TN'][currRepeat] = bonfTNTot
                    except IndexError:
                        multiplerepeats.Bonferroni['TN'] = np.append(multiplerepeats.Bonferroni['TN'], bonfTNTot)
                    try:
                        multiplerepeats.Bonferroni['FN'][currRepeat] = bonfFNTot
                    except IndexError:
                        multiplerepeats.Bonferroni['FN'] = np.append(multiplerepeats.Bonferroni['FN'], bonfFNTot)
                    try:
                        multiplerepeats.Bonferroni['FD'][currRepeat] = bonfFDTot
                    except IndexError:
                        multiplerepeats.Bonferroni['FD'] = np.append(multiplerepeats.Bonferroni['FD'], bonfFDTot)
                    
                    try:
                        multiplerepeats.BenjHoch['TP'][currRepeat] = bhTPTot
                    except IndexError:
                        multiplerepeats.BenjHoch['TP'] = np.append(multiplerepeats.BenjHoch['TP'], bhTPTot)
                    try:
                        multiplerepeats.BenjHoch['FP'][currRepeat] = bhFPTot
                    except IndexError:
                        multiplerepeats.BenjHoch['FP'] = np.append(multiplerepeats.BenjHoch['FP'], bhFPTot)
                    try:
                        multiplerepeats.BenjHoch['TN'][currRepeat] = bhTNTot
                    except IndexError:
                        multiplerepeats.BenjHoch['TN'] = np.append(multiplerepeats.BenjHoch['TN'], bhTNTot)
                    try:                    
                        multiplerepeats.BenjHoch['FN'][currRepeat] = bhFNTot
                    except IndexError:
                        multiplerepeats.BenjHoch['FN'] = np.append(multiplerepeats.BenjHoch['FN'], bhFNTot)
                    try:
                        multiplerepeats.BenjHoch['FD'][currRepeat] = bhFDTot
                    except IndexError:
                        multiplerepeats.BenjHoch['FD'] = np.append(multiplerepeats.BenjHoch['FD'], bhFDTot)
                    
                    try:
                        multiplerepeats.BenjYek['TP'][currRepeat] = byTPTot
                    except IndexError:
                        multiplerepeats.BenjYek['TP']= np.append(multiplerepeats.BenjYek['TP'], byTPTot)
                    try:
                        multiplerepeats.BenjYek['FP'][currRepeat] = byFPTot
                    except IndexError:
                        multiplerepeats.BenjYek['FP'] = np.append(multiplerepeats.BenjYek['FP'], byFPTot)
                    try:
                        multiplerepeats.BenjYek['TN'][currRepeat] = byTNTot
                    except IndexError:
                        multiplerepeats.BenjYek['TN'] = np.append(multiplerepeats.BenjYek['TN'], byTNTot)
                    try:
                        multiplerepeats.BenjYek['FN'][currRepeat] = byFNTot
                    except IndexError:
                        multiplerepeats.BenjYek['FN'] = np.append(multiplerepeats.BenjYek['FN'], byFNTot)
                    try:
                        multiplerepeats.BenjYek['FD'][currRepeat] = byFDTot
                    except IndexError:
                        multiplerepeats.BenjYek['FD'] = np.append(multiplerepeats.BenjYek['FD'], byFDTot)
                ##end of for currRepeat in range(0, nRepeats):
                    
                    #get multiplerepeats.Bonferroni keys/fields
                stats = []
                for key, value in multiplerepeats.Bonferroni.iteritems():
                    stats.append(key)
                for currstat in stats:
                    uncStruct[currstat][currEff][currSampSize] = np.mean(multiplerepeats.noCorrection[currstat])
                    uncStruct['S'+currstat][currEff][currSampSize] = np.std(multiplerepeats.noCorrection[currstat])
                        
                    bonfStruct[currstat][currEff][currSampSize] = np.mean(multiplerepeats.Bonferroni[currstat])
                    bonfStruct['S'+currstat][currEff][currSampSize] = np.std(multiplerepeats.Bonferroni[currstat])
                        
                    byStruct[currstat][currEff][currSampSize] = np.mean(multiplerepeats.BenjYek[currstat])
                    byStruct['S'+currstat][currEff][currSampSize] = np.std(multiplerepeats.BenjYek[currstat])
                        
                    bhStruct[currstat][currEff][currSampSize] = np.mean(multiplerepeats.BenjHoch[currstat])
                    bhStruct['S'+currstat][currEff][currSampSize] = np.std(multiplerepeats.BenjHoch[currstat])
                        
        ## end of for currEff in range(1,nEffSizes+1):
        stats = []
        for key, value in uncStruct.iteritems():
                        stats.append(key)
        for i in range(0, len(stats)):
            try:
                storeVar[0][i] = uncStruct[stats[i]] 
            except IndexError:
                if (nEffSizes == 1 and nSampSizes == 1):
                    storeVar = np.append(storeVar, np.zeros((4,1)), axis =1)                
                elif (nEffSizes > 1 or nSampSizes > 1):
                    storeVar = np.append(storeVar, np.zeros((4,1, nEffSizes, nSampSizes)), axis=1)                
                storeVar[0][i] = uncStruct[stats[i]]            
            try:
                storeVar[1][i] = bonfStruct[stats[i]] 
            except IndexError:
                if (nEffSizes == 1 and nSampSizes == 1):
                    storeVar = np.append(storeVar, np.zeros((4,1)), axis =1)                
                elif (nEffSizes > 1 or nSampSizes > 1):
                    storeVar = np.append(storeVar, np.zeros((4,1, nEffSizes, nSampSizes)), axis=1)                
                storeVar[1][i] = bonfStruct[stats[i]]            
            try:
                storeVar[2][i] = bhStruct[stats[i]] 
            except IndexError:
                if (nEffSizes == 1 and nSampSizes == 1):
                    storeVar = np.append(storeVar, np.zeros((4,1)), axis =1)                
                elif (nEffSizes > 1 or nSampSizes > 1):
                    storeVar = np.append(storeVar, np.zeros((4,1, nEffSizes, nSampSizes)), axis=1)                
                storeVar[2][i] = bhStruct[stats[i]]
                        
            try:
                storeVar[3][i] = byStruct[stats[i]] 
            except IndexError:
                if (nEffSizes == 1 and nSampSizes == 1):
                    storeVar = np.append(storeVar, np.zeros((4,1)), axis =1)                
                elif (nEffSizes > 1 or nSampSizes > 1):
                    storeVar = np.append(storeVar, np.zeros((4,1, nEffSizes, nSampSizes)), axis=1)                
                storeVar[2][i] = byStruct[stats[i]]
                
        output.append(storeVar)
        ## storeVar1 = np.expand_dims(storeVar, axis=0)
        ## try:
            ## output=np.append(output,storeVar1,axis=0)
        ## except ValueError:
            ## output=storeVar1
        ## print output.shape
                  
    #output2[currVar].append(output)        
    print '|| \n'
    try:
        return output
    except:
        print 'error occurs when returning output in parallel'
    
def randperm1(totalLen):
    #function of random permuation and pick up the sub array according to the specified size
    tempList = np.random.permutation(totalLen)                  ##generate a random permutation array
    return tempList
##=======End of PCalc_Continuous====================

##=======Beginning of PCalc_2Group====================
def PCalc_2Group(data, EffectSizes, SampSizes, SignThreshold, nSimSamp, nRepeat):
    ##If sample size bigger than number of simulated samples adjust it
    ## global sampSizes, signThreshold, effectSizes, numVars, nRepeats, nSampSizes, nEffSizes, Samples_seg, correlationMat_seg, output2
    ## global output2
    sampSizes = SampSizes
    signThreshold = SignThreshold
    effectSizes = EffectSizes
    
    try:
        if 2*max(sampSizes) >= nSimSamp:
            print 'Number of simulated samples smaller than maximum of samplesizes to check - increased'
            nSimSamp = 2*max(sampSizes) + 500
    except ValueError:
        if 2*max(max(sampSizes)) >= nSimSamp:
            print 'Number of simulated samples smaller than maximum of samplesizes to check - increased'
            nSimSamp = 2*max(max(sampSizes)) + 500
    ## convert matrix to numpy array type if needed
    if (type(data).__name__ != 'ndarray'):
        data = np.array(data)
    
    ##get data array size
    size = data.shape
    rows = size[0]
    if (data.ndim > 1):
        cols = size[1]    
        
    ##Number of variables
    numVars = cols

    nRepeats = nRepeat
    
    ##Number of sample and effect sizes
    if (sampSizes.ndim >1):
        nSampSizes = sampSizes.shape[1]
    elif (sampSizes.ndim ==1):
        nSampSizes = sampSizes.shape[0]
    if (effectSizes.ndim >1):
        nEffSizes = effectSizes.shape[1]
    elif (effectSizes.ndim ==1):
        nEffSizes = effectSizes.shape[0]
    

    ##Simulation of a new data set based on multivariate normal distribution
    Samples, correlationMat = simulateLogNormal(data,'Estimate', nSimSamp)
        
    ##split Samples and correlationMat into chunk files for parallel processing
    if multiprocessing.cpu_count()-1 <= 0:
        cores = 1
    else:
        cores = multiprocessing.cpu_count()
        
    Samples_seg = _chunkMatrix(Samples, cores)
    correlationMat_seg = _chunkMatrix(correlationMat, cores)
    
    output2=[]
    ## for ii in range(cores):
        ## output2.append([])  
        
    output2 = Parallel(n_jobs=cores)(delayed(f_multiproc)(sampSizes, signThreshold, effectSizes, numVars, nRepeats, nSampSizes, nEffSizes, Samples_seg, correlationMat_seg, ii) for ii in range(cores))            #n_jobs=-1 means using all CPUs or any other number>1 to specify the number of CPUs to use
    ## for ii in range(numVars):
            ## f_multiproc(ii)
    ##pass the results to output
    output = np.array([])
    output = np.array(output2[0])
    for ii in range(1, cores):
        output = np.append(output, output2[ii],axis=0)
        
    try:
        return output
    except:
        print 'error occurs when returning output'
        
def f_multiproc(sampSizes, signThreshold, effectSizes, numVars, nRepeats, nSampSizes, nEffSizes, Samples_seg, correlationMat_seg, currCore):
    ## global Samples_seg, correlationMat_seg, output2, nEffSizes, nSampSizes, nRepeats, sampSizes, effectSizes
    ## global output2 # this definition doesn't work

    #re-check numVars
    numVars = Samples_seg[currCore].shape[1]
    
    output=[]
    
    if (nEffSizes == 1 and nSampSizes == 1):
        storeVar = np.zeros((4,nRepeats))
    elif (nEffSizes > 1 or nSampSizes > 1):
        storeVar = np.zeros((4,nRepeats, nEffSizes, nSampSizes))                            
    #define uncStruct -- structual data
    uncStruct = {'TP':np.zeros((nEffSizes, nSampSizes)),'FP':np.zeros((nEffSizes, nSampSizes)),'TN':np.zeros((nEffSizes, nSampSizes)),\
                 'FN':np.zeros((nEffSizes, nSampSizes)),'FD':np.zeros((nEffSizes, nSampSizes)),'STP':np.zeros((nEffSizes, nSampSizes)),\
                 'SFP':np.zeros((nEffSizes, nSampSizes)),'STN':np.zeros((nEffSizes, nSampSizes)),'SFN':np.zeros((nEffSizes, nSampSizes)),\
                 'SFD':np.zeros((nEffSizes, nSampSizes))}
    
    bonfStruct = {'TP':np.zeros((nEffSizes, nSampSizes)),'FP':np.zeros((nEffSizes, nSampSizes)),'TN':np.zeros((nEffSizes, nSampSizes)),\
                 'FN':np.zeros((nEffSizes, nSampSizes)),'FD':np.zeros((nEffSizes, nSampSizes)),'STP':np.zeros((nEffSizes, nSampSizes)),\
                 'SFP':np.zeros((nEffSizes, nSampSizes)),'STN':np.zeros((nEffSizes, nSampSizes)),'SFN':np.zeros((nEffSizes, nSampSizes)),\
                 'SFD':np.zeros((nEffSizes, nSampSizes))}
    
    bhStruct = {'TP':np.zeros((nEffSizes, nSampSizes)),'FP':np.zeros((nEffSizes, nSampSizes)),'TN':np.zeros((nEffSizes, nSampSizes)),\
                 'FN':np.zeros((nEffSizes, nSampSizes)),'FD':np.zeros((nEffSizes, nSampSizes)),'STP':np.zeros((nEffSizes, nSampSizes)),\
                 'SFP':np.zeros((nEffSizes, nSampSizes)),'STN':np.zeros((nEffSizes, nSampSizes)),'SFN':np.zeros((nEffSizes, nSampSizes)),\
                 'SFD':np.zeros((nEffSizes, nSampSizes))}
    
    byStruct = {'TP':np.zeros((nEffSizes, nSampSizes)),'FP':np.zeros((nEffSizes, nSampSizes)),'TN':np.zeros((nEffSizes, nSampSizes)),\
                 'FN':np.zeros((nEffSizes, nSampSizes)),'FD':np.zeros((nEffSizes, nSampSizes)),'STP':np.zeros((nEffSizes, nSampSizes)),\
                 'SFP':np.zeros((nEffSizes, nSampSizes)),'STN':np.zeros((nEffSizes, nSampSizes)),'SFN':np.zeros((nEffSizes, nSampSizes)),\
                 'SFD':np.zeros((nEffSizes, nSampSizes))}
    for currVar in range(0,numVars):
        for currEff in range(0,nEffSizes):
            for currSampSize in range(0,nSampSizes):
                # define the structural data multiplerepeats
                class MUltiplerepeats(object):
                    def __init__(self,Results,Bonferroni,BenjHoch,BenjYek,noCorrection):
                        self.Results = Results
                        self.Bonferroni = Bonferroni
                        self.BenjHoch = BenjHoch
                        self.BenjYek = BenjYek
                        self.noCorrection = noCorrection
                multiplerepeats=MUltiplerepeats({'TP':np.zeros(nRepeats),'FP':np.zeros(nRepeats),'TN':np.zeros(nRepeats),'FN':np.zeros(nRepeats),'FD':np.zeros(nRepeats)},\
                                                {'TP':np.zeros(nRepeats),'FP':np.zeros(nRepeats),'TN':np.zeros(nRepeats),'FN':np.zeros(nRepeats),'FD':np.zeros(nRepeats)},\
                                                {'TP':np.zeros(nRepeats),'FP':np.zeros(nRepeats),'TN':np.zeros(nRepeats),'FN':np.zeros(nRepeats),'FD':np.zeros(nRepeats)},\
                                                {'TP':np.zeros(nRepeats),'FP':np.zeros(nRepeats),'TN':np.zeros(nRepeats),'FN':np.zeros(nRepeats),'FD':np.zeros(nRepeats)},\
                                                {'TP':np.zeros(nRepeats),'FP':np.zeros(nRepeats),'TN':np.zeros(nRepeats),'FN':np.zeros(nRepeats),'FD':np.zeros(nRepeats)})
                
                for currRepeat in range(0, nRepeats):
                    ## Select a subset of the simulated spectra
                    selectIndex = randperm(len(Samples_seg[currCore]), 2 * sampSizes[0][currSampSize])
                                    
                    
                    if (type(selectIndex).__name__ != 'ndarray'):
                        selectIndex = np.array(selectIndex)
                    SelSamples = Samples_seg[currCore][selectIndex]                    # matrix slicing
                    
                    
                    ##Assume class balanced, modify proportion of group here
                    GroupId = np.ones((len(SelSamples),1))
                    for i in range(int(floor(len(SelSamples)/2)), len(SelSamples)):
                        GroupId[i][0] = 2
                        
                    ##Introduce change
                    corrVector = np.array([])
                    corrVector = correlationMat_seg[currCore][:,currVar]
                    
                        
                    ## stdSelSamples = np.std(SelSamples, axis=0, ddof=1)
                    for k in range(0,numVars):
                        if (corrVector[k]>0.8):
                            for j in range(0, len(GroupId)):
                                if (GroupId[j][0]==2):
                                    stdSelSamples = np.std(SelSamples, axis=0, ddof=1)
                                    SelSamples[j][k] = SelSamples[j][k] + effectSizes[0][currEff]*stdSelSamples[k]
    
                    ##Initialize p value vector for this round
                                        
                    p = np.zeros((1,numVars))
                    for var2check in range(0,numVars):
                        tempSamples1 = []
                        tempSamples2 = []
                        for i in range(0, len(SelSamples)):
                            if (GroupId[i][0]==1):
                                tempSamples1.append(SelSamples[i][var2check])
                            if (GroupId[i][0]==2):
                                tempSamples2.append(SelSamples[i][var2check])  
                        p[0][var2check] = scistats.f_oneway(tempSamples1,tempSamples2)[1]
                        
                        
                    pUnc = p                ##pUnc and p have 1xnumVars elements
                    pBonf = p * numVars     ##pBonf has 1xnumVars elements
                    
                    
                    h1, crit_p, adj_ci_cvrg, pBY = fdr_bh(p, 0.05, 'dep')
                    h1, crit_p, adj_ci_cvrg, pBH = fdr_bh(p, 0.05, 'pdep')
                    
                                    
                    uncTNTot, uncTPTot, uncFPTot, uncFNTot, uncFDTot = calcConfMatrixUniv(pUnc, corrVector, signThreshold, 0.8)
                    bonfTNTot, bonfTPTot, bonfFPTot, bonfFNTot, bonfFDTot = calcConfMatrixUniv(pBonf, corrVector, signThreshold, 0.8)
                    
                    byTNTot, byTPTot, byFPTot, byFNTot, byFDTot = calcConfMatrixUniv(pBY, corrVector, signThreshold, 0.8)
                    bhTNTot, bhTPTot, bhFPTot, bhFNTot, bhFDTot = calcConfMatrixUniv(pBH, corrVector, signThreshold, 0.8)
                    
                    try:
                        multiplerepeats.noCorrection['TP'][currRepeat] = uncTPTot
                    except IndexError:
                        multiplerepeats.noCorrection['TP'] = np.append(multiplerepeats.noCorrection['TP'], uncTPTot)  #if array index exceeds upper bound, extend the array
                    try:    
                        multiplerepeats.noCorrection['FP'][currRepeat] = uncFPTot
                    except IndexError:
                        multiplerepeats.noCorrection['FP'] = np.append(multiplerepeats.noCorrection['FP'], uncFPTot)
                    try:
                        multiplerepeats.noCorrection['TN'][currRepeat] = uncTNTot
                    except IndexError:
                        multiplerepeats.noCorrection['TN'] = np.append(multiplerepeats.noCorrection['TN'], uncTNTot)
                    try:
                        multiplerepeats.noCorrection['FN'][currRepeat] = uncFNTot
                    except IndexError:
                        multiplerepeats.noCorrection['FN'] = np.append(multiplerepeats.noCorrection['FN'], uncFNTot)
                    try:
                        multiplerepeats.noCorrection['FD'][currRepeat] = uncFDTot
                    except IndexError:
                        multiplerepeats.noCorrection['FD'] = np.append(multiplerepeats.noCorrection['FD'], uncFDTot)
                        
                        
                    try:
                        multiplerepeats.Bonferroni['TP'][currRepeat] = bonfTPTot
                    except IndexError:
                        multiplerepeats.Bonferroni['TP'] = np.append(multiplerepeats.Bonferroni['TP'], bonfTPTot)
                    try:
                        multiplerepeats.Bonferroni['FP'][currRepeat] = bonfFPTot
                    except IndexError:
                        multiplerepeats.Bonferroni['FP'] = np.append(multiplerepeats.Bonferroni['FP'], bonfFPTot)
                    try:
                        multiplerepeats.Bonferroni['TN'][currRepeat] = bonfTNTot
                    except IndexError:
                        multiplerepeats.Bonferroni['TN'] = np.append(multiplerepeats.Bonferroni['TN'], bonfTNTot)
                    try:
                        multiplerepeats.Bonferroni['FN'][currRepeat] = bonfFNTot
                    except IndexError:
                        multiplerepeats.Bonferroni['FN'] = np.append(multiplerepeats.Bonferroni['FN'], bonfFNTot)
                    try:
                        multiplerepeats.Bonferroni['FD'][currRepeat] = bonfFDTot
                    except IndexError:
                        multiplerepeats.Bonferroni['FD'] = np.append(multiplerepeats.Bonferroni['FD'], bonfFDTot)
                    
                    try:
                        multiplerepeats.BenjHoch['TP'][currRepeat] = bhTPTot
                    except IndexError:
                        multiplerepeats.BenjHoch['TP'] = np.append(multiplerepeats.BenjHoch['TP'], bhTPTot)
                    try:
                        multiplerepeats.BenjHoch['FP'][currRepeat] = bhFPTot
                    except IndexError:
                        multiplerepeats.BenjHoch['FP'] = np.append(multiplerepeats.BenjHoch['FP'], bhFPTot)
                    try:
                        multiplerepeats.BenjHoch['TN'][currRepeat] = bhTNTot
                    except IndexError:
                        multiplerepeats.BenjHoch['TN'] = np.append(multiplerepeats.BenjHoch['TN'], bhTNTot)
                    try:                    
                        multiplerepeats.BenjHoch['FN'][currRepeat] = bhFNTot
                    except IndexError:
                        multiplerepeats.BenjHoch['FN'] = np.append(multiplerepeats.BenjHoch['FN'], bhFNTot)
                    try:
                        multiplerepeats.BenjHoch['FD'][currRepeat] = bhFDTot
                    except IndexError:
                        multiplerepeats.BenjHoch['FD'] = np.append(multiplerepeats.BenjHoch['FD'], bhFDTot)
                    
                    try:
                        multiplerepeats.BenjYek['TP'][currRepeat] = byTPTot
                    except IndexError:
                        multiplerepeats.BenjYek['TP']= np.append(multiplerepeats.BenjYek['TP'], byTPTot)
                    try:
                        multiplerepeats.BenjYek['FP'][currRepeat] = byFPTot
                    except IndexError:
                        multiplerepeats.BenjYek['FP'] = np.append(multiplerepeats.BenjYek['FP'], byFPTot)
                    try:
                        multiplerepeats.BenjYek['TN'][currRepeat] = byTNTot
                    except IndexError:
                        multiplerepeats.BenjYek['TN'] = np.append(multiplerepeats.BenjYek['TN'], byTNTot)
                    try:
                        multiplerepeats.BenjYek['FN'][currRepeat] = byFNTot
                    except IndexError:
                        multiplerepeats.BenjYek['FN'] = np.append(multiplerepeats.BenjYek['FN'], byFNTot)
                    try:
                        multiplerepeats.BenjYek['FD'][currRepeat] = byFDTot
                    except IndexError:
                        multiplerepeats.BenjYek['FD'] = np.append(multiplerepeats.BenjYek['FD'], byFDTot)
                ##end of for currRepeat in range(0, nRepeats):
                    
                    #get multiplerepeats.Bonferroni keys/fields
                stats = []
                for key, value in multiplerepeats.Bonferroni.iteritems():
                    stats.append(key)
                for currstat in stats:
                    uncStruct[currstat][currEff][currSampSize] = np.mean(multiplerepeats.noCorrection[currstat])
                    uncStruct['S'+currstat][currEff][currSampSize] = np.std(multiplerepeats.noCorrection[currstat])
                        
                    bonfStruct[currstat][currEff][currSampSize] = np.mean(multiplerepeats.Bonferroni[currstat])
                    bonfStruct['S'+currstat][currEff][currSampSize] = np.std(multiplerepeats.Bonferroni[currstat])
                        
                    byStruct[currstat][currEff][currSampSize] = np.mean(multiplerepeats.BenjYek[currstat])
                    byStruct['S'+currstat][currEff][currSampSize] = np.std(multiplerepeats.BenjYek[currstat])
                        
                    bhStruct[currstat][currEff][currSampSize] = np.mean(multiplerepeats.BenjHoch[currstat])
                    bhStruct['S'+currstat][currEff][currSampSize] = np.std(multiplerepeats.BenjHoch[currstat])
                        
        ## end of for currEff in range(0,nEffSizes):
        stats = []
        for key, value in uncStruct.iteritems():
                        stats.append(key)
        for i in range(0, len(stats)):
            try:
                storeVar[0][i] = uncStruct[stats[i]] 
            except IndexError:
                if (nEffSizes == 1 and nSampSizes == 1):
                    storeVar = np.append(storeVar, np.zeros((4,1)), axis =1)                
                elif (nEffSizes > 1 or nSampSizes > 1):
                    storeVar = np.append(storeVar, np.zeros((4,1, nEffSizes, nSampSizes)), axis=1)                
                storeVar[0][i] = uncStruct[stats[i]]
                
            try:
                storeVar[1][i] = bonfStruct[stats[i]] 
            except IndexError:
                if (nEffSizes == 1 and nSampSizes == 1):
                    storeVar = np.append(storeVar, np.zeros((4,1)), axis =1)                
                elif (nEffSizes > 1 or nSampSizes > 1):
                    storeVar = np.append(storeVar, np.zeros((4,1, nEffSizes, nSampSizes)), axis=1)                
                storeVar[1][i] = bonfStruct[stats[i]]
                
            try:
                storeVar[2][i] = bhStruct[stats[i]] 
            except IndexError:
                if (nEffSizes == 1 and nSampSizes == 1):
                    storeVar = np.append(storeVar, np.zeros((4,1)), axis =1)                
                elif (nEffSizes > 1 or nSampSizes > 1):
                    storeVar = np.append(storeVar, np.zeros((4,1, nEffSizes, nSampSizes)), axis=1)                
                storeVar[2][i] = bhStruct[stats[i]]
                
            try:
                storeVar[3][i] = byStruct[stats[i]] 
            except IndexError:
                if (nEffSizes == 1 and nSampSizes == 1):
                    storeVar = np.append(storeVar, np.zeros((4,1)), axis =1)                
                elif (nEffSizes > 1 or nSampSizes > 1):
                    storeVar = np.append(storeVar, np.zeros((4,1, nEffSizes, nSampSizes)), axis=1)                
                storeVar[2][i] = byStruct[stats[i]]
            
        output.append(storeVar)
    print '| \n'
    try:        
        return output
    except:        
        print 'error occurs when returning output in parallel'         
    
    
    
def _chunkMatrix(data, num): ##different from Caroline's one, which uses list
    cols = data.shape[1]
    avg = int(ceil(cols / float(num)))
    ##out = np.zeros((num, data.shape[0], avg))
    out = []
    for i in range(num):
        out.append([])
    for i in range(0, num-1):
        out[i] = data[:, np.array(range(i*avg, i*avg+avg))]
    last = int(num-1)
    out[last] = data[:, np.array(range(last*avg, cols))]       
    return out

def randperm(totalLen, subLen):
    #function of random permuation and pick up the sub array according to the specified size
    tempList = np.random.permutation(totalLen)                  ##generate a random permutation array
    tempList1 = random.sample(tempList,subLen)
    return tempList1

def write_file(data,filename): #creates file and writes list to it
  np.savetxt(filename, data, delimiter=",")
##=======End of PCalc_2Group====================

## function of false discovery rate Benjamini & Hochberg FDR_BH
## function expecting is for detecting the number of outputs; written by Sami Hangaslammi
def expecting():
    """Return how many values the caller is expecting"""
    f = inspect.currentframe()
    f = f.f_back.f_back
    c = f.f_code
    i = f.f_lasti
    bytecode = c.co_code
    instruction = ord(bytecode[i+3])
    if instruction == dis.opmap['UNPACK_SEQUENCE']:
        howmany = ord(bytecode[i+4])
        return howmany
    elif instruction == dis.opmap['POP_TOP']:
        return 0
    return 1

def fdr_bh(*args):
    try:
        pvals = args[0]  
        ##convert to numpy array type if not the ndarray type
        if (type(pvals).__name__ != 'ndarray'): 
            pvals = np.array(pvals)               
    except IndexError:
      print "Usage: fdr_bh(<arg1>,<arg2>,<arg3>,<arg4>)"
      print "arg1 as p-value matrix (mandatory must be provided"
      print "arg2 as false discovery rate(optional)"
      print "arg3 as method:'pdep' or 'dep', 'pdep' is given as default(optional)"
      print "arg4 as report:'yes' or 'no', 'no' is given as default(optional)"
      sys.exit(1)
          
    if len(args)<2:
        q = 0.05
    else:
        q = args[1]    
    if len(args)<3:
        method = 'pdep'
    else:
        method = args[2]
    if len(args)<4:
        report = 'no'
    else:
        report = args[3]
        
    s = pvals.shape
    if (pvals.ndim > 1):                                                #if pvals has more than 1 rows, reshape into 1 row array
        reshaped_pvals = np.reshape(pvals, (1, np.prod(s)))
        p_sorted = np.sort(reshaped_pvals)
        sort_ids = np.argsort(reshaped_pvals)
    else:                                                               # pvals is already 1xn array
        p_sorted = np.sort(pvals)
        sort_ids = np.argsort(pvals)
    dummy = np.sort(sort_ids)
    unsort_ids = np.argsort(sort_ids)
    
    if (type(p_sorted[0]).__name__ == 'ndarray'):
        m = len(p_sorted[0])
    else:
        m = len(p_sorted)
        
    if (method == 'pdep'):
        #BH procedure for independence or positive dependence
        thresh=np.arange(1,m+1)*q/m
        wtd_p=m*p_sorted/np.arange(1,m+1)   
    elif (method == 'dep'):
        #BH procedure for any dependency structure
        denom=m*sum(1.0/np.arange(1,m+1))
        thresh=np.arange(1,m+1)*q/denom
        wtd_p=denom*p_sorted/np.arange(1,m+1)
        '''
        Note, it can produce adjusted p-values greater than 1!
        compute adjusted p-values
        '''
    else:
        print 'Argument \'method\' needs to be \'pdep\' or \'dep\'.'
    
    nargout = expecting()                       #get the number of expecting outputs from caller
    if (nargout > 3):
        #compute adjusted p-values
        adj_p=np.zeros(m)*float('NaN')
        wtd_p_sorted = np.sort(wtd_p)        
        wtd_p_sindex = np.argsort(wtd_p)
        nextfill = 0
        for k in range(0,m):
            if (wtd_p_sindex[0][k]>=nextfill):
                adj_p[nextfill:(wtd_p_sindex[0][k]+1)] = wtd_p_sorted[0][k]
                nextfill = wtd_p_sindex[0][k]+1
                if (nextfill>m):
                    break
        adj_p=np.reshape(adj_p[unsort_ids],s)
        
    rej=p_sorted<=thresh
    
    try:
        max_id=max(max(np.where(rej[0] == True)))                       #find greatest significant pvalue
    except ValueError:
        max_id=max(np.where(rej[0] == True))
    if not max_id:
        # if the max_id is empty
        crit_p=0
        h1=pvals*0
        adj_ci_cvrg=float('NaN')
    else:
        crit_p=p_sorted[0][max_id]
        h1=pvals<=crit_p
        adj_ci_cvrg=1-thresh[max_id]
            
    if (report == 'yes'):
        n_sig=sum(p_sorted<=crit_p)
        if (n_sig==1):
            print 'Out of %d tests, %d is significant using a false discovery rate of %f.\n' %(m,n_sig,q)
        else:
            print 'Out of %d tests, %d are significant using a false discovery rate of %f.\n'%(m,n_sig,q)
        if (method == 'pdep'):
            print 'FDR/FCR procedure used is guaranteed valid for independent or positively dependent tests.\n'
        else:
            print 'FDR/FCR procedure used is guaranteed valid for independent or dependent tests.\n'
    ## return the results
    try:
        return h1, crit_p, adj_ci_cvrg, adj_p
    except:
        print "Errors occur when returning h1, crit_p, adj_ci_cvrg and adj_p"
        
        
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

def read2array(filename):
    dataArray = []
    try:
        with open(filename) as infile:
            for line in infile:
                dataArray.append(line.strip().split(','))
        dataArray = [[float(x) for x in y] for y in dataArray]              #The array was created with all elements as strings. Convert into floats.
        dataArray = np.array(dataArray)                                     #convert to numpy array type
    except IOError:
        print filename + " does not exist!"
        
    return dataArray
def main():    
    ## read the data into an array;
    XSRV = read2array('paTest.csv')
    
    ##print array size
    if (XSRV.ndim > 1):
        rows = XSRV.shape[0]
        cols = XSRV.shape[1]
    elif (XSRV.ndim == 1):
        rows = 1
        cols = XSRV.shape[0]
    
    print 'Input data matrix size is :' + str(rows) + ',' + str(cols)
    ## Part I
    ##Run code for a single effect and sample size combination as a test
    ## effectSizes = np.array([[0.5]])
    ## sampleSizes = np.array([[200]])
    ## numberreps = 10
    
    ## diffgroups = np.array([])
    ## linearregression = np.array([])
    ## diffgroups = PCalc_2Group(XSRV,effectSizes, sampleSizes, 0.05, 5000, numberreps)
    ## linearregression = PCalc_Continuous(XSRV,effectSizes, sampleSizes, 0.05, 5000, numberreps)
    
    ## Part II 
    ##Define a grid of effect sizes and sample sizes to test
    effectSizes = np.array([[0.05, 0.1, 0.15,0.2, 0.25, 0.3, 0.35]])
    sampleSizes = np.array([[50, 100, 200, 250, 350, 500, 750, 1000]])
    numberreps= 10
    ## ## Calculat for a subset of 4 variables (less than 20 seconds on 4-core desktop for each analysis)
    diffgroups = np.array([])
    linearregression = np.array([])
    t_start = datetime.now()
    diffgroups = PCalc_2Group(XSRV[:,np.arange(0,8)],effectSizes, sampleSizes, 0.05, 5000, numberreps);
    linearregression = PCalc_Continuous(XSRV[:,np.arange(0,8)],effectSizes, sampleSizes, 0.05, 5000, numberreps)
    t_end = datetime.now()
    print 'Part II A -time collapsed: ' + str(t_end-t_start)
    ## ## ## Surface plot function (see details in bottom of tutorial)
    ## ## SurfacePlot(diffgroups, 2, 4,2 , sampleSizes, effectSizes,numberreps)


    ## ## Run the code for all variables. Each analysis takes around 1h on a 4 core desktop. To speed up, use less effect and sample 
    ## ## sample sizes and a smaller number of repeats
    ## t_start = datetime.now()
    ## diffgroups = PCalc_2Group(XSRV,effectSizes, sampleSizes, 0.05, 5000, numberreps)
    ## linearregression = PCalc_Continuous(XSRV,effectSizes, sampleSizes, 0.05, 5000, numberreps)
    ## t_end = datetime.now()
    ## print 'Part II B -time collapsed: ' + str(t_end-t_start)

    '''
    Using the SurfacePlot function to visualize results 
    SurfacePlot(output, variable,metric,correction, sizeeff,samplsizes,nreps)
    Output is the structure returned from the simulator, variable is the index of variable to plot
    metric is the to display and correction the type of multiple testing correction to 
    visualize.
    
    Metric options:
    1 - True positive Rate
    2 - False Positive Rate
    3 - True Negative Rate
    4 - False Negative Rate
    Correction:
    1 - No correction
    2 - Bonferroni
    3 - Benjamini-Hochberg
    4 - Benjamini-Yekutieli
    
    The example line below will open the False Negative Rate surface for
    variable number 2 without multiple testing correction
    '''
    SurfacePlot(diffgroups, 2, 4,2 , sampleSizes, effectSizes,numberreps)
              
if __name__=="__main__":
    main()