##this is the main function
import os
import csv
import numpy as np
from datetime import datetime
from PCalc_2Group import PCalc_2Group
from PCalc_Continuous import PCalc_Continuous
from SurfacePlot import SurfacePlot

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
    XSRV = read2array('TutorialData.csv')
    
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
    diffgroups = PCalc_2Group(XSRV[:,np.arange(0,108)],effectSizes, sampleSizes, 0.05, 5000, numberreps);
    linearregression = PCalc_Continuous(XSRV[:,np.arange(0,108)],effectSizes, sampleSizes, 0.05, 5000, numberreps)
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