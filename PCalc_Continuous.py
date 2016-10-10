##function of PCalc_Continuous
import sys, csv, os.path, datetime
from joblib import Parallel, delayed                    #for Parallel computing
import random
import numpy as np
import statsmodels.formula.api as sm                    #for linear regression
from math import ceil
from simulateLogNormal import simulateLogNormal
from fdr_bh import fdr_bh
from calcConfMatrixUniv import calcConfMatrixUniv
# Set up multiprocessing enviroment
import multiprocessing

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
    
    output2 = Parallel(n_jobs=cores)(delayed(f_multiproc)(sampSizes, signThreshold, effectSizes, nRepeats, nSampSizes, nEffSizes, Samples_seg, correlationMat_seg,ii) for ii in range(cores))            #n_jobs=-1 means using all CPUs or any other number>1 to specify the number of CPUs to use
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

def f_multiproc(sampSizes, signThreshold, effectSizes, nRepeats, nSampSizes, nEffSizes, Samples_seg, correlationMat_seg, currCore):
    
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
def randperm1(totalLen):
    #function of random permuation and pick up the sub array according to the specified size
    tempList = np.random.permutation(totalLen)                  ##generate a random permutation array
    return tempList

def write_file(data,filename): #creates file and writes list to it
    np.savetxt(filename, data, delimiter=",")
    
    
if __name__=="__main__":
    PCalc_Continuous(data, EffectSizes, SampSizes, SignThreshold, nSimSamp, nRepeat)			# run the main function