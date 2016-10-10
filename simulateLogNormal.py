## function of simulateLogNormal
import sys
import numpy as np
from math import log,fabs,exp
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
    
if __name__=="__main__":
    simulateLogNormal(data, covType, nSamples)			# run the main function