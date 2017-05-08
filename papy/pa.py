#!/usr/bin/env python
"""
Power Analysis (calculation) tool
Developed by Dr. Goncalo Correia and Dr Jianliang Gao
Imperial College London
2016
"""
import os,sys,csv,inspect,dis,os.path,random,multiprocessing,getopt
import numpy as np
import scipy.stats as scistats
import statsmodels.formula.api as sm                    #for linear regression
import shutil                                           #for creating zip files
from math import fabs,floor,ceil,log,exp
from datetime import datetime
##--not use following modules anymore
##import matplotlib.pyplot as plt
##from joblib import Parallel, delayed                    #for Parallel computing
##from statsmodels import robust                          #for work out median absolute deviation

## For 3d plots. This import is necessary to have 3D plotting below
##from mpl_toolkits.mplot3d import Axes3D
## for saving the plot to pdf file 
##from matplotlib.backends.backend_pdf import PdfPages

##=======Beginning of interactive SurfacePlot============
def iSurfacePlot(output, svfilename, variable,metric,correction, sizeeff,samplsizes,nreps):
    import plotly as py
    import plotly.graph_objs as go
    MUtot = output[variable-1][correction-1][metric-1]
    NS, NSE = MUtot.shape
    ##SIGMAtot = output[variable-1][correction-1][metric+5-1]
    ##SIGMAlow=MUtot-1.96*SIGMAtot/np.sqrt(nreps)
    ##SIGMAlow = np.array([[0 if x<0 else x for x in y] for y in SIGMAlow])
    
    ##plot
    ##generate a 2D grid
    X, Y = np.meshgrid(sizeeff, samplsizes)
    
    
    zaxis_title = 'True Positive Rate'
    if metric == 1:
        if not 'mean' in svfilename:
            zaxis_title = 'True Positive Rate'
        else:
            if 'tp' in svfilename:
                zaxis_title = 'True Positive Rate'
            if 'fp' in svfilename:
                zaxis_title = 'False Positive Rate'
            if 'tn' in svfilename:
                zaxis_title = 'True Negative Rate'
            if 'fn' in svfilename:
                zaxis_title = 'False Negative Rate'
    elif metric == 2:
        zaxis_title = 'False Positive Rate'
    elif metric == 3:
        zaxis_title = 'True Negative Rate'
    elif metric == 4:
        zaxis_title = 'False Negative Rate'
    camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=2, y=2, z=0.1)
            )    
    layout = go.Layout(
        title='Statistical Power Analysis Resutls',
        autosize=True,
        width=1024,
        height=768,
        margin=go.Margin(
            l=80,
            r=40,
            b=100,
            t=60
        ),
        scene=go.Scene(
            xaxis=dict(
                title='Sample Sizes',
                range=[0,np.max(X)+0.1]
            ),
            yaxis=dict(
                title='Effect Sizes',
                tickmode='linear',
                tick0=0,
                dtick=0.1,
                range=[0,np.max(Y)]
            ),
            zaxis=dict(
                title=zaxis_title,
                tickmode='linear',
                tick0=0,
                dtick=0.1,
                range=[0,1.0]
            )
        )
    )
    data=[go.Surface(x=X,y=Y,z=MUtot)]
    fig = go.Figure(data=data, layout=layout)
    fig['layout'].update(scene=dict(camera=camera))
    py.offline.plot(fig, filename=svfilename, auto_open=False)
##=======End of interactive SurfacePlot============

##====== Beginning of scatter plot for slices of surface plots===============
def iSlicesPlot(X, Y, Error_y, svfilename, plot_title, x_caption, y_caption, trace_label, trace_num):
    import plotly as py
    import plotly.graph_objs as go
    
    traces = []
    for ii in range(0, len(Y)):
        trace_tmp = go.Scatter(x=X,y=Y[ii], error_y=dict(
                type='data',
                array=Error_y[ii],
                visible=True
                ),
                name=trace_label+str(trace_num[0][ii])
            )
        traces.append(trace_tmp)
        
    data=go.Data(traces)
    
    ##define other features of plots
    
    ##dictionary of y_caption
    if y_caption == 'tpn':
        y_caption = 'True Positive Rate'
        plot_title = plot_title+'-(no correction)'
    if y_caption == 'tpb':
        y_caption = 'True Positive Rate'
        plot_title = plot_title+'-(Bonferroni correction)'
    if y_caption == 'tpbh':
        y_caption = 'True Positive Rate'
        plot_title = plot_title+'-(Benjamini-Hochberg correction)'
    if y_caption == 'tpby':
        y_caption = 'True Positive Rate'
        plot_title = plot_title+'-(Benjamini-Yekutieli correction)'
    
    if y_caption == 'fpn':
        y_caption = 'False Positive Rate'
        plot_title = plot_title+'-(no correction)'
    if y_caption == 'fpb':
        y_caption = 'False Positive Rate'
        plot_title = plot_title+'-(Bonferroni correction)'
    if y_caption == 'fpbh':
        y_caption = 'False Positive Rate'
        plot_title = plot_title+'-(Benjamini-Hochberg correction)'
    if y_caption == 'fpby':
        y_caption = 'False Positive Rate'
        plot_title = plot_title+'-(Benjamini-Yekutieli correction)'    
    
    if y_caption == 'tnn':
        y_caption = 'True Negative Rate'
        plot_title = plot_title+'-(no correction)'
    if y_caption == 'tnb':
        y_caption = 'True Negative Rate'
        plot_title = plot_title+'-(Bonferroni correction)'
    if y_caption == 'tnbh':
        y_caption = 'True Negative Rate'
        plot_title = plot_title+'-(Benjamini-Hochberg correction)'
    if y_caption == 'tnby':
        y_caption = 'True Negative Rate'
        plot_title = plot_title+'-(Benjamini-Yekutieli correction)'
    
    if y_caption == 'fnn':
        y_caption = 'False Negative Rate'
        plot_title = plot_title+'-(no correction)'
    if y_caption == 'fnb':
        y_caption = 'False Negative Rate'
        plot_title = plot_title+'-(Bonferroni correction)'
    if y_caption == 'fnbh':
        y_caption = 'False Negative Rate'
        plot_title = plot_title+'-(Benjamini-Hochberg correction)'
    if y_caption == 'fnby':
        y_caption = 'False Negative Rate'
        plot_title = plot_title+'-(Benjamini-Yekutieli correction)'

    layout = go.Layout(
        title= plot_title,
        xaxis=dict(
            title=x_caption,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title=y_caption,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
)
    fig = go.Figure(data=data, layout=layout)
    py.offline.plot(fig, filename = svfilename, auto_open=False)
##====== End of scatter plot for slices of surface plots===============

##====== Beginning of surface plot for power rate only===============
def iSurfacePlotTPR(output, svfilename, correction, sizeeff,samplsizes,nreps):
    import plotly as py
    import plotly.graph_objs as go
    MUtot = output
    NS, NSE = MUtot.shape
        
    ##plot
    ##generate a 2D grid
    X, Y = np.meshgrid(sizeeff, samplsizes)
    
    ##define z axis title
    zaxis_title = 'Proportion'
    
    camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=2, y=2, z=0.1)
            )    
    layout = go.Layout(
        title='Proportion of Variables with Power (True Positive)> 0.8 -%s, %d Repeats'%(correction, nreps),
        autosize=True,
        width=1024,
        height=768,
        margin=go.Margin(
            l=80,
            r=40,
            b=100,
            t=60
        ),
        scene=go.Scene(
            xaxis=dict(
                title='Sample Sizes',
                range=[0,np.max(X)+0.1]
            ),
            yaxis=dict(
                title='Effect Sizes',
                tickmode='linear',
                tick0=0,
                dtick=0.1,
                range=[0,np.max(Y)]
            ),
            zaxis=dict(
                title=zaxis_title,
                tickmode='linear',
                tick0=0,
                dtick=0.1,
                range=[0,1.0]
            )
        )
    )
    data=[go.Surface(x=X,y=Y,z=MUtot)]
    fig = go.Figure(data=data, layout=layout)
    fig['layout'].update(scene=dict(camera=camera))
    py.offline.plot(fig, filename=svfilename, auto_open=False)
    
##====== End of surface plot for power rate only===============    
    

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
        print('Unknown Covariance type')
    
    ##np.random.seed(10)                                  ##add random seed for testing purpose    
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
def PCalc_Continuous(data, EffectSizes, SampSizes, SignThreshold, nSimSamp, nRepeat,cores):
    ##If sample size bigger than number of simulated samples adjust it
    sampSizes = SampSizes
    signThreshold = SignThreshold
    effectSizes = EffectSizes
    
    try:
        if max(sampSizes) >= nSimSamp:
            print('Number of simulated samples smaller than maximum of samplesizes to check - increased')
            nSimSamp = max(sampSizes) + 500
    except ValueError:
        if max(max(sampSizes)) >= nSimSamp:
            print('Number of simulated samples smaller than maximum of samplesizes to check - increased')
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
    Samples_seg = Samples
    correlationMat_seg = correlationMat
    
    ## Initialize the output structures
    output2=[]
    
    ##define an array for storing the results in each step of repeat for all variables
    ##with all effect sizes and sample sizes; 
    output_allsteps_tmp=[]
    
    ## output2 = Parallel(n_jobs=cores)(delayed(f_multiproc1)(sampSizes, signThreshold, effectSizes, nRepeats, nSampSizes, nEffSizes, Samples_seg, correlationMat_seg, cols, cores, ii) for ii in range(cores))            #n_jobs=-1 means using all CPUs or any other number>1 to specify the number of CPUs to use
    ## for ii in range(numVars):       # non parallelized loop
        ## f_multiproc(ii)
    pool = multiprocessing.Pool(processes=cores)
    output2 = [pool.apply_async(f_multiproc,args=(sampSizes, signThreshold, effectSizes, numVars, nRepeats, nSampSizes, nEffSizes, Samples_seg, correlationMat_seg, cols, cores, wk)) for wk in range(cores)]
    output2 = [p.get(None) for p in output2]
    output3 = list()
    # Unpack results
    for instanceOutput in output2:
        for item in instanceOutput:
            output3.append(item)
    
    ##pass the results to output
    output = []
    
    ##work out number of overall results and number of power rate results
    num_overall_results = int(round(numVars / cores))
    for novr in range(num_overall_results):
        output.append(output3[novr])
    
    ##pass Power TPR results to output variables
    output_uncTP = []
    output_bonfTP = []
    output_bhTP = []
    output_byTP = []
    
    output_uncTP = np.array(output3[num_overall_results])
    output_bonfTP = np.array(output3[num_overall_results+1])
    output_bhTP = np.array(output3[num_overall_results+2])
    output_byTP = np.array(output3[num_overall_results+3])
    
    if cores>1:
        for ii in range(1, cores-1):
            for novr in range(num_overall_results):
                output.append(output3[ii*(num_overall_results+4)+novr])
            
            output_uncTP = np.append(output_uncTP, output3[ii*(num_overall_results+4)+num_overall_results],axis=2)
            output_bonfTP = np.append(output_bonfTP, output3[ii*(num_overall_results+4)+num_overall_results+1],axis=2)
            output_bhTP = np.append(output_bhTP, output3[ii*(num_overall_results+4)+num_overall_results+2],axis=2)
            output_byTP = np.append(output_byTP, output3[ii*(num_overall_results+4)+num_overall_results+3],axis=2)
        
        rest_num_overall_results = numVars - num_overall_results * (cores-1)
        for novr in range(rest_num_overall_results):
            output.append(output3[(cores-1)*(num_overall_results+4)+novr]) 
        output_uncTP = np.append(output_uncTP, output3[(cores-1)*(num_overall_results+4)+rest_num_overall_results],axis=2)
        output_bonfTP = np.append(output_bonfTP, output3[(cores-1)*(num_overall_results+4)+rest_num_overall_results+1],axis=2)
        output_bhTP = np.append(output_bhTP, output3[(cores-1)*(num_overall_results+4)+rest_num_overall_results+2],axis=2)
        output_byTP = np.append(output_byTP, output3[(cores-1)*(num_overall_results+4)+rest_num_overall_results+3],axis=2)        
    
    output = np.array(output)    
    ##for the mean proportion of number of variables achieve the power; and the std
    output_uncTP_ratio_median = np.zeros((nEffSizes, nSampSizes))
    output_bonfTP_ratio_median = np.zeros((nEffSizes, nSampSizes))
    output_bhTP_ratio_median = np.zeros((nEffSizes, nSampSizes))
    output_byTP_ratio_median = np.zeros((nEffSizes, nSampSizes))
    
    output_uncTP_ratio_iqr = np.zeros((nEffSizes, nSampSizes))
    output_bonfTP_ratio_iqr = np.zeros((nEffSizes, nSampSizes))
    output_bhTP_ratio_iqr = np.zeros((nEffSizes, nSampSizes))
    output_byTP_ratio_iqr = np.zeros((nEffSizes, nSampSizes))
    
    for currEff in range(0, nEffSizes):
        for currSamp in range(0, nSampSizes):
            tmp_median_array = np.zeros(nRepeats)
            for currStep in range(0, nRepeats):
                tmp_median_array[currStep] = (sum(1 for x in output_uncTP[currEff][currSamp][:,currStep] if x>0.8))/float(numVars)
            output_uncTP_ratio_median[currEff][currSamp] = np.median(tmp_median_array)
            try:
                output_uncTP_ratio_iqr[currEff][currSamp] = scistats.iqr(tmp_median_array, axis=0, interpolation='midpoint')
            except AttributeError:
                q75, q25 = np.percentile(tmp_median_array, [75, 25], axis=0)
                output_uncTP_ratio_iqr[currEff][currSamp] = q75-q25


            
            tmp_median_array = np.zeros(nRepeats)
            for currStep in range(0, nRepeats):
                tmp_median_array[currStep] = (sum(1 for x in output_bonfTP[currEff][currSamp][:,currStep] if x>0.8))/float(numVars)
            output_bonfTP_ratio_median[currEff][currSamp] = np.median(tmp_median_array)
            try:
                output_bonfTP_ratio_iqr[currEff][currSamp] = scistats.iqr(tmp_median_array, axis=0, interpolation='midpoint')
            except AttributeError:
                q75, q25 = np.percentile(tmp_median_array, [75, 25], axis=0)
                output_bonfTP_ratio_iqr[currEff][currSamp] = q75-q25


            
            tmp_median_array = np.zeros(nRepeats)
            for currStep in range(0, nRepeats):
                tmp_median_array[currStep] = (sum(1 for x in output_bhTP[currEff][currSamp][:,currStep] if x>0.8))/float(numVars)
            output_bhTP_ratio_median[currEff][currSamp] = np.median(tmp_median_array)
            try:
                output_bhTP_ratio_iqr[currEff][currSamp] = scistats.iqr(tmp_median_array, axis=0, interpolation='midpoint')
            except AttributeError:
                q75, q25 = np.percentile(tmp_median_array, [75, 25], axis=0)
                output_bhTP_ratio_iqr[currEff][currSamp] = q75-q25

            
            tmp_median_array = np.zeros(nRepeats)
            for currStep in range(0, nRepeats):
                tmp_median_array[currStep] = (sum(1 for x in output_byTP[currEff][currSamp][:,currStep] if x>0.8))/float(numVars)            
            output_byTP_ratio_median[currEff][currSamp] = np.median(tmp_median_array)
            try:
                output_byTP_ratio_iqr[currEff][currSamp] = scistats.iqr(tmp_median_array, axis=0, interpolation='midpoint')
            except AttributeError:
                q75, q25 = np.percentile(tmp_median_array, [75, 25], axis=0)
                output_byTP_ratio_iqr[currEff][currSamp] = q75-q25

          
    try:
        return output, output_uncTP_ratio_median, output_bonfTP_ratio_median, output_bhTP_ratio_median, output_byTP_ratio_median,\
                output_uncTP_ratio_iqr, output_bonfTP_ratio_iqr, output_bhTP_ratio_iqr, output_byTP_ratio_iqr
    except:
        print('error occurs when returning output')

def f_multiproc1(sampSizes, signThreshold, effectSizes, nRepeats, nSampSizes, nEffSizes, Samples_seg, correlationMat_seg, cols, cores, currCore):
    
    ##re-check numVars
    offSet = currCore*int(round(cols/cores))
    if (currCore<(cores-1)):
        numVars = int(round(cols/cores))
    else:
        numVars = cols - int(round(cols/cores))*(cores-1)
    
    ##for storing all results in all repeated steps with all effect sizes and sample
    ##sizes for Power (TP) in current samples_seg
    output_all_uncTP_tmp=np.zeros((nEffSizes, nSampSizes, numVars, nRepeats))
    output_all_bonfTP_tmp=np.zeros((nEffSizes, nSampSizes, numVars, nRepeats))
    output_all_bhTP_tmp=np.zeros((nEffSizes, nSampSizes, numVars, nRepeats))
    output_all_byTP_tmp=np.zeros((nEffSizes, nSampSizes, numVars, nRepeats))
    
    ##for storing results of all metric and correction options under the combination
    ##of effect size and sample size for all variables
    output = []
    
    
        
    ##define uncStruct -- dictionary data
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
        if (nEffSizes == 1 and nSampSizes == 1):
            storeVar = np.zeros((4,nRepeats))
        elif (nEffSizes > 1 or nSampSizes > 1):
            storeVar = np.zeros((4,nRepeats, nEffSizes, nSampSizes))
            
        for currEff in range(0,nEffSizes):
            b1 = np.zeros((numVars,1))
            b1[currVar][0] = effectSizes[0][currEff]
            
            for currSampSize in range(0,nSampSizes):
                ## define the structural data multiplerepeats
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
                    SelSamples = Samples_seg[selectIndex]                    # matrix slicing
                                    
                    ## UVScaling the data - vectorize with bsxfun 
                    stDev = np.std(SelSamples, axis=0)   # without argument ddof=1 means using default ddof=0 to work out std on population
                    SelSamples = SelSamples - np.mean(SelSamples, axis=0)
                    SelSamples =  SelSamples/stDev
                    
                    noiseLevel = 1
                    
                    noise = noiseLevel*np.random.randn(sampSizes[0][currSampSize],1)
                    
                    Y = SelSamples[:, np.array([currVar])]*b1[currVar][0]
                    Y = Y + noise
                                        
                    p = np.zeros((1,numVars))
                    
                    ##Using regress for multivariate regression test
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
                    
                    ##
                    corrVector = correlationMat_seg[:,currVar+offSet]
                    
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
                        
                ##storing each step of repeats    
                output_all_uncTP_tmp[currEff][currSampSize][currVar]=multiplerepeats.noCorrection['TP']
                output_all_bonfTP_tmp[currEff][currSampSize][currVar]=multiplerepeats.Bonferroni['TP']
                output_all_bhTP_tmp[currEff][currSampSize][currVar]=multiplerepeats.BenjYek['TP']
                output_all_byTP_tmp[currEff][currSampSize][currVar]=multiplerepeats.BenjYek['TP']    
                    
                ##end of for currRepeat in range(0, nRepeats):
                    
                ##get multiplerepeats.Bonferroni keys/fields
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
                storeVar[3][i] = byStruct[stats[i]]
                
        output.append(storeVar)

    print('|| \n')
    output.append(output_all_uncTP_tmp)
    output.append(output_all_bonfTP_tmp)
    output.append(output_all_bhTP_tmp)
    output.append(output_all_byTP_tmp)
    try:
        return output
    except:
        print('error occurs when returning output in parallel')
    
def randperm1(totalLen):
    ##function of random permuation and pick up the sub array according to the specified size
    tempList = np.random.permutation(totalLen)                  ##generate a random permutation array
    return tempList
##=======End of PCalc_Continuous====================

##=======Beginning of PCalc_2Group====================
def PCalc_2Group(data, EffectSizes, SampSizes, SignThreshold, nSimSamp, nRepeat,cores):
    ##If sample size bigger than number of simulated samples adjust it
    ## global sampSizes, signThreshold, effectSizes, numVars, nRepeats, nSampSizes, nEffSizes, Samples_seg, correlationMat_seg, output2
    ## global output2
    sampSizes = SampSizes
    signThreshold = SignThreshold
    effectSizes = EffectSizes
    
    try:
        if 2*max(sampSizes) >= nSimSamp:
            print('Number of simulated samples smaller than maximum of samplesizes to check - increased')
            nSimSamp = 2*max(sampSizes) + 500
    except ValueError:
        if 2*max(max(sampSizes)) >= nSimSamp:
            print('Number of simulated samples smaller than maximum of samplesizes to check - increased')
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
    
    Samples_seg = Samples
    correlationMat_seg=correlationMat
    output2=[]
    
    ##define an array for storing the results in each step of repeat for all variables
    ##with all effect sizes and sample sizes; 
    output_allsteps_tmp=[]
    ## #debug - using another multiporcessing method to run f_multiproc
    pool = multiprocessing.Pool(processes=cores)
    output2 = [pool.apply_async(f_multiproc,args=(sampSizes, signThreshold, effectSizes, numVars, nRepeats, nSampSizes, nEffSizes, Samples_seg, correlationMat_seg, cols, cores, wk)) for wk in range(cores)]
    output2 = [p.get(None) for p in output2]
    output3 = list()
    # Unpack results
    for instanceOutput in output2:
        for item in instanceOutput:
            output3.append(item)
    
    ## output2 = Parallel(n_jobs=cores)(delayed(f_multiproc)(sampSizes, signThreshold, effectSizes, numVars, nRepeats, nSampSizes, nEffSizes, Samples_seg, correlationMat_seg, cols, cores, ii) for ii in range(cores))            #n_jobs=-1 means using all CPUs or any other number>1 to specify the number of CPUs to use
    ## for ii in range(numVars): ##for non-parallel running
            ## f_multiproc(ii)
    ##pass the results to output
    output = []
    
    ##work out number of overall results and number of power rate results
    num_overall_results = int(round(numVars / cores))
    for novr in range(num_overall_results):
        output.append(output3[novr])
    
    ##pass Power TPR results to output variables
    output_uncTP = []
    output_bonfTP = []
    output_bhTP = []
    output_byTP = []
    
    output_uncTP = np.array(output3[num_overall_results])
    output_bonfTP = np.array(output3[num_overall_results+1])
    output_bhTP = np.array(output3[num_overall_results+2])
    output_byTP = np.array(output3[num_overall_results+3])
    
    if cores>1:
        for ii in range(1, cores-1):
            for novr in range(num_overall_results):
                output.append(output3[ii*(num_overall_results+4)+novr])
            
            output_uncTP = np.append(output_uncTP, output3[ii*(num_overall_results+4)+num_overall_results],axis=2)
            output_bonfTP = np.append(output_bonfTP, output3[ii*(num_overall_results+4)+num_overall_results+1],axis=2)
            output_bhTP = np.append(output_bhTP, output3[ii*(num_overall_results+4)+num_overall_results+2],axis=2)
            output_byTP = np.append(output_byTP, output3[ii*(num_overall_results+4)+num_overall_results+3],axis=2)
        
        rest_num_overall_results = numVars - num_overall_results * (cores-1)
        for novr in range(rest_num_overall_results):
            output.append(output3[(cores-1)*(num_overall_results+4)+novr]) 
        output_uncTP = np.append(output_uncTP, output3[(cores-1)*(num_overall_results+4)+rest_num_overall_results],axis=2)
        output_bonfTP = np.append(output_bonfTP, output3[(cores-1)*(num_overall_results+4)+rest_num_overall_results+1],axis=2)
        output_bhTP = np.append(output_bhTP, output3[(cores-1)*(num_overall_results+4)+rest_num_overall_results+2],axis=2)
        output_byTP = np.append(output_byTP, output3[(cores-1)*(num_overall_results+4)+rest_num_overall_results+3],axis=2)
    
    
    output = np.array(output)    
    ##for the mean proportion of number of variables achieve the power; and the std
    output_uncTP_ratio_median = np.zeros((nEffSizes, nSampSizes))
    output_bonfTP_ratio_median = np.zeros((nEffSizes, nSampSizes))
    output_bhTP_ratio_median = np.zeros((nEffSizes, nSampSizes))
    output_byTP_ratio_median = np.zeros((nEffSizes, nSampSizes))
    
    output_uncTP_ratio_iqr = np.zeros((nEffSizes, nSampSizes))
    output_bonfTP_ratio_iqr = np.zeros((nEffSizes, nSampSizes))
    output_bhTP_ratio_iqr = np.zeros((nEffSizes, nSampSizes))
    output_byTP_ratio_iqr = np.zeros((nEffSizes, nSampSizes))
    
    for currEff in range(0, nEffSizes):
        for currSamp in range(0, nSampSizes):
            tmp_median_array = np.zeros(nRepeats)
            for currStep in range(0, nRepeats):
                tmp_median_array[currStep] = (sum(1 for x in output_uncTP[currEff][currSamp][:,currStep] if x>0.8))/float(numVars)
            output_uncTP_ratio_median[currEff][currSamp] = np.median(tmp_median_array)
            try:
                output_uncTP_ratio_iqr[currEff][currSamp] = scistats.iqr(tmp_median_array, axis=0, interpolation='midpoint')
            except AttributeError:
                q75, q25 = np.percentile(tmp_median_array, [75, 25], axis=0)
                output_uncTP_ratio_iqr[currEff][currSamp] = q75-q25
            
            tmp_median_array = np.zeros(nRepeats)
            for currStep in range(0, nRepeats):
                tmp_median_array[currStep] = (sum(1 for x in output_bonfTP[currEff][currSamp][:,currStep] if x>0.8))/float(numVars)
            output_bonfTP_ratio_median[currEff][currSamp] = np.median(tmp_median_array)
            try:
                output_bonfTP_ratio_iqr[currEff][currSamp] = scistats.iqr(tmp_median_array, axis=0, interpolation='midpoint')
            except AttributeError:
                q75, q25 = np.percentile(tmp_median_array, [75, 25], axis=0)
                output_bonfTP_ratio_iqr[currEff][currSamp] = q75-q25

            
            tmp_median_array = np.zeros(nRepeats)
            for currStep in range(0, nRepeats):
                tmp_median_array[currStep] = (sum(1 for x in output_bhTP[currEff][currSamp][:,currStep] if x>0.8))/float(numVars)
            output_bhTP_ratio_median[currEff][currSamp] = np.median(tmp_median_array)
            try:
                output_bhTP_ratio_iqr[currEff][currSamp] = scistats.iqr(tmp_median_array, axis=0, interpolation='midpoint')
            except AttributeError:
                q75, q25 = np.percentile(tmp_median_array, [75, 25], axis=0)
                output_bhTP_ratio_iqr[currEff][currSamp] = q75-q25

            
            tmp_median_array = np.zeros(nRepeats)
            for currStep in range(0, nRepeats):
                tmp_median_array[currStep] = (sum(1 for x in output_byTP[currEff][currSamp][:,currStep] if x>0.8))/float(numVars)            
            output_byTP_ratio_median[currEff][currSamp] = np.median(tmp_median_array)
            try:
                output_byTP_ratio_iqr[currEff][currSamp] = scistats.iqr(tmp_median_array, axis=0, interpolation='midpoint')
            except AttributeError:
                q75, q25 = np.percentile(tmp_median_array, [75, 25], axis=0)
                output_byTP_ratio_iqr[currEff][currSamp] = q75-q25
    try:
        return output, output_uncTP_ratio_median, output_bonfTP_ratio_median, output_bhTP_ratio_median, output_byTP_ratio_median,\
                output_uncTP_ratio_iqr, output_bonfTP_ratio_iqr, output_bhTP_ratio_iqr, output_byTP_ratio_iqr, \
                output_uncTP, output_bonfTP, output_bhTP, output_byTP
            
    except:
        print('error occurs when returning output')
        
def f_multiproc(sampSizes, signThreshold, effectSizes, numVars, nRepeats, nSampSizes, nEffSizes, Samples_seg, correlationMat_seg, cols, cores, currCore):
    
    ##re-check numVars
    offSet = currCore*int(round(cols/cores))
    if (currCore<(cores-1)):
        numVars = int(round(cols/cores))
    else:
        numVars = cols - int(round(cols/cores))*(cores-1)
    
    #debug
    print("numVars=%d; current core=%d"%(numVars, currCore))
    ##for storing all results in all repeated steps with all effect sizes and sample
    ##sizes for Power (TP) in current samples_seg
    output_all_uncTP_tmp=np.zeros((nEffSizes, nSampSizes, numVars, nRepeats))
    output_all_bonfTP_tmp=np.zeros((nEffSizes, nSampSizes, numVars, nRepeats))
    output_all_bhTP_tmp=np.zeros((nEffSizes, nSampSizes, numVars, nRepeats))
    output_all_byTP_tmp=np.zeros((nEffSizes, nSampSizes, numVars, nRepeats))
    
    output=[]
    
    if (nEffSizes == 1 and nSampSizes == 1):
        storeVar = np.zeros((4,10))
    elif (nEffSizes > 1 or nSampSizes > 1):
        storeVar = np.zeros((4,10, nEffSizes, nSampSizes))                            
    ##define uncStruct, bonfStruct, bhStruct, byStruct  -- dictionary data 
    ## the key's order, if retrieving by index, is FP,TN,FD,FN,SFD,STP,STN,SFN,SFP,TP
    ##STP-- State for True Positive prediction; SFP -- State for False Positive prediction
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
        if (nEffSizes == 1 and nSampSizes == 1):
            storeVar = np.zeros((4,10))
        elif (nEffSizes > 1 or nSampSizes > 1):
            storeVar = np.zeros((4,10, nEffSizes, nSampSizes))
            
        for currEff in range(0,nEffSizes):
            for currSampSize in range(0,nSampSizes):
                ## define the structural data multiplerepeats
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
                    selectIndex = randperm(len(Samples_seg), 2 * sampSizes[0][currSampSize])  ##sampSizes is a 1xn array
                    
                    ## check selectIndex is a numpy array, if not, then convert to numpy array.
                    if (type(selectIndex).__name__ != 'ndarray'):
                        selectIndex = np.array(selectIndex)
                    SelSamples = Samples_seg[selectIndex]                    # matrix slicing by rows
                    
                    
                    ##Assume class balanced, modify proportion of group here
                    GroupId = np.ones((len(SelSamples),1))
                    for i in range(int(floor(len(SelSamples)/2)), len(SelSamples)):
                        GroupId[i][0] = 2
                        
                    ##Introduce change
                    corrVector = np.array([])
                    #corrVector = correlationMat_seg[currCore][:,currVar] ##this line caused error of calculation on 2nd or other cores
                    corrVector = correlationMat_seg[:,currVar+offSet]
                
                        
                    stdSelSamples = np.std(SelSamples, axis=0, ddof=1)
                    for k in range(0,cols):
                        if (corrVector[k]>0.8):
                            for j in range(0, len(GroupId)):
                                if (GroupId[j][0]==2):
                                    ## stdSelSamples = np.std(SelSamples, axis=0, ddof=1)
                                    SelSamples[j][k] = SelSamples[j][k] + effectSizes[0][currEff]*stdSelSamples[k]
    
                    ##Initialize p value vector for this round
                                        
                    p = np.zeros((1,cols))
                    for var2check in range(0,cols):
                        p[0][var2check] = scistats.f_oneway(SelSamples[0:int(len(SelSamples)/2),var2check],SelSamples[int(len(SelSamples)/2):len(SelSamples),var2check])[1]
                        ## tempSamples1 = []
                        ## tempSamples2 = []
                        ## for i in range(0, len(SelSamples)):
                            ## if (GroupId[i][0]==1):
                                ## tempSamples1.append(SelSamples[i][var2check])
                            ## if (GroupId[i][0]==2):
                                ## tempSamples2.append(SelSamples[i][var2check])  
                        ## p[0][var2check] = scistats.f_oneway(tempSamples1,tempSamples2)[1]
                        
                        
                    pUnc = p                ##pUnc and p have 1xnumVars elements
                    pBonf = p * cols     ##pBonf has 1xnumVars elements
                    
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
                        
                output_all_uncTP_tmp[currEff][currSampSize][currVar]=multiplerepeats.noCorrection['TP']
                output_all_bonfTP_tmp[currEff][currSampSize][currVar]=multiplerepeats.Bonferroni['TP']
                output_all_bhTP_tmp[currEff][currSampSize][currVar]=multiplerepeats.BenjYek['TP']
                output_all_byTP_tmp[currEff][currSampSize][currVar]=multiplerepeats.BenjYek['TP']
                    
                ##get multiplerepeats.Bonferroni keys/fields
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
                storeVar[3][i] = byStruct[stats[i]]
            
        output.append(storeVar)
    print('| \n')
    output.append(output_all_uncTP_tmp)
    output.append(output_all_bonfTP_tmp)
    output.append(output_all_bhTP_tmp)
    output.append(output_all_byTP_tmp)
    
    try:        
        return output
    except:        
        print('error occurs when returning output in parallel')      
    
    
def _chunkMatrix(data, num): ##different from Caroline's one, which uses list
    cols = data.shape[1]
    avg = int(round(cols / float(num)))
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
    ##function of random permuation and pick up the sub array according to the specified size
    ##np.random.seed(10)                                  ##add random seed for testing purpose
    tempList = np.random.permutation(totalLen)                  ##generate a random permutation array
    ##random.seed(10)                                  ##add random seed for testing purpose
    try:
        tempList1 = random.sample(tempList,subLen)
    except TypeError:
        tempList1 = random.sample(list(tempList),subLen)
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
      print("Usage: fdr_bh(<arg1>,<arg2>,<arg3>,<arg4>)")
      print("arg1 as p-value matrix (mandatory must be provided")
      print("arg2 as false discovery rate(optional)")
      print("arg3 as method:'pdep' or 'dep', 'pdep' is given as default(optional)")
      print("arg4 as report:'yes' or 'no', 'no' is given as default(optional)")
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
        ##BH procedure for independence or positive dependence
        thresh=np.arange(1,m+1)*q/m
        wtd_p=m*p_sorted/np.arange(1,m+1)   
    elif (method == 'dep'):
        ##BH procedure for any dependency structure
        denom=m*sum(1.0/np.arange(1,m+1))
        thresh=np.arange(1,m+1)*q/denom
        wtd_p=denom*p_sorted/np.arange(1,m+1)
        '''
        Note, it can produce adjusted p-values greater than 1!
        compute adjusted p-values
        '''
    else:
        print('Argument \'method\' needs to be \'pdep\' or \'dep\'.')
    
    nargout = expecting()                       #get the number of expecting outputs from caller
    if (nargout > 3):
        ##compute adjusted p-values
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
        ## if the max_id is empty
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
            print('Out of %d tests, %d is significant using a false discovery rate of %f.\n' %(m,n_sig,q))
        else:
            print('Out of %d tests, %d are significant using a false discovery rate of %f.\n'%(m,n_sig,q))
        if (method == 'pdep'):
            print('FDR/FCR procedure used is guaranteed valid for independent or positively dependent tests.\n')
        else:
            print('FDR/FCR procedure used is guaranteed valid for independent or dependent tests.\n')
    ## return the results
    try:
        return h1, crit_p, adj_ci_cvrg, adj_p
    except:
        print("Errors occur when returning h1, crit_p, adj_ci_cvrg and adj_p")
        
        
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
        #debugging
        #print "fabsfabs(corrVector[%d])=%f; Pf[0][%d]=%r"%(i,fabs(corrVector[i]),i, Pf[0][i])
        if ((fabs(corrVector[i]) < corrThresh) and (Pf[0][i]==False)):
            TN=TN+1
            #print "TN=%d"%(TN)
        elif ((fabs(corrVector[i]) > corrThresh) and (Pf[0][i]==True)):
            TP=TP+1
            #print "TP=%d"%(TP)
        elif ((fabs(corrVector[i]) > corrThresh) and (Pf[0][i]==False)):
            FN=FN+1
            #print "FN=%d"%(FN)
        elif ((fabs(corrVector[i]) < corrThresh) and (Pf[0][i]==True)):
            FP=FP+1
            #print "FP=%d"%(FP)
        
    try:
        TNtot = TN/(FP+TN)
    except ZeroDivisionError:
        TNtot = float('NaN')
    try: #TPR - power
        TPtot = TP/(TP+FN)
    except ZeroDivisionError:
        TPtot = float('NaN')
        #TPtot = 0.0
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
        print("Errors occur when returning uncTNTot, uncTPTot, uncFPTot, uncFNTot, uncFDTot")

def read2array(filename):
    dataArray = []
    try:
        with open(filename) as infile:
            for line in infile:
                dataArray.append(line.strip().split(','))
        dataArray = [[float(x) for x in y] for y in dataArray]              #The array was created with all elements as strings. Convert into floats.
        dataArray = np.array(dataArray)                                     #convert to numpy array type
    except IOError:
        print(filename + " does not exist!")
        
    return dataArray


def main(argv1, argv2, argv3, argv4, argv5, argv6): 
    
    ## read the data into an array;
    XSRV = read2array(argv1)
    if (type(XSRV).__name__ != 'ndarray'):
        XSRV = np.array(XSRV)
    ##print array size
    if (XSRV.ndim > 1):
        rows = XSRV.shape[0]
        cols = XSRV.shape[1]
    elif (XSRV.ndim == 1):
        rows = 1
        cols = XSRV.shape[0]
    
    print('Input data matrix size is :' + str(rows) + ',' + str(cols))
    
    tmpStr=argv2.split('-')
    if len(tmpStr)>1:
        argv2=[int(tmpStr[0]),int(tmpStr[1])+1]
    else:
        argv2=[0,int(argv2)]
    #debugging
    print(argv2[0],argv2[1])
    
    tmpStr=argv3.split(':')
    argv3=range(int(tmpStr[0]), int(tmpStr[2]), int(tmpStr[1]))
    if argv3[0]==0:
        argv3[0]=1
    argv3=np.array(argv3)
    argv3=np.reshape(argv3,(1,len(argv3))) 
        
    tmpStr=argv4.split(':')
    argv4=np.arange(float(tmpStr[0]), float(tmpStr[2]), float(tmpStr[1]))
    if argv4[0]==0:
        argv4[0]=1
    argv4=np.array(argv4)
    argv4=np.reshape(argv4,(1,len(argv4))) 

    sampleSizes = argv3 #np.array([[1, 50, 100, 200, 250, 350, 500, 750, 1000]])
    effectSizes = argv4 #np.array([[0.05, 0.1, 0.15,0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]])
    
    
    ##define output metric options
    metric_opt = np.array([1, 2, 3, 4])  #see options description below
    correction_opt = np.array([1, 2, 3, 4]) #see correction options description below
    
    
    numberreps= int(argv5)
    
    cores = int(argv6)

    ## ## Calculat for a subset of 4 variables (less than 20 seconds on 4-core desktop for each analysis)
    diffgroups = np.array([])
    linearregression = np.array([])
    t_start = datetime.now()
    num_cols = int(argv2[1])-int(argv2[0])
    ##if the number of variables is less than the request CPU cores, use number of variables as cores.
    if (num_cols<cores):
        cores=num_cols
        
    if (num_cols > 0):
        diffgroups, output_uncTP_ratio_median, output_bonfTP_ratio_median, output_bhTP_ratio_median, output_byTP_ratio_median,\
                output_uncTP_ratio_iqr, output_bonfTP_ratio_iqr, output_bhTP_ratio_iqr, output_byTP_ratio_iqr, \
                output_uncTP, output_bonfTP, output_bhTP, output_byTP \
                = PCalc_2Group(XSRV[:,np.arange(int(argv2[0]), int(argv2[1]))],effectSizes, sampleSizes, 0.05, 5000, numberreps, cores)
        linearregression, output_uncTP_ratio_median_ln, output_bonfTP_ratio_median_ln, output_bhTP_ratio_median_ln, output_byTP_ratio_median_ln,\
                output_uncTP_ratio_iqr_ln, output_bonfTP_ratio_iqr_ln, output_bhTP_ratio_iqr_ln, output_byTP_ratio_iqr_ln \
                 = PCalc_Continuous(XSRV[:,np.arange(int(argv2[0]), int(argv2[1]))],effectSizes, sampleSizes, 0.05, 5000, numberreps, cores)
        t_end = datetime.now()
        print('Time elapsed: ' + str(t_end-t_start))
   
    else:
        t_start = datetime.now()
        diffgroups, output_uncTP_ratio_median, output_bonfTP_ratio_median, output_bhTP_ratio_median, output_byTP_ratio_median,\
                output_uncTP_ratio_iqr, output_bonfTP_ratio_iqr, output_bhTP_ratio_iqr, output_byTP_ratio_iqr, \
                output_uncTP, output_bonfTP, output_bhTP, output_byTP \
                = PCalc_2Group(XSRV,effectSizes, sampleSizes, 0.05, 5000, numberreps, cores)
        linearregression, output_uncTP_ratio_median_ln, output_bonfTP_ratio_median_ln, output_bhTP_ratio_median_ln, output_byTP_ratio_median_ln,\
                output_uncTP_ratio_iqr_ln, output_bonfTP_ratio_iqr_ln, output_bhTP_ratio_iqr_ln, output_byTP_ratio_iqr_ln \
                = PCalc_Continuous(XSRV,effectSizes, sampleSizes, 0.05, 5000, numberreps, cores)
        t_end = datetime.now()
        print('Time elapsed: ' + str(t_end-t_start))

    ##diffgroups has dimension of (number of variables, 4, 10, effectsize, samplesize);
    ##number of variables is the input number of columns from the input dataset.
    ##4-- 4 correction options
    ##10--10 metric as "TP","FP","TN","FN","FD","STP","SFP","STN","SFN","SFD" 

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
    ##write diffgroups and linearregression into file for testing purpose
    ##np.savetxt('diffgroups.csv',diffgroups[1][3][1], delimiter=",")
    ##np.savetxt('linearregression.csv',linearregression[1][3][1], delimiter=",")
    if not os.path.exists('papy_output'):
        os.makedirs('papy_output')
    
    ##file names matrix
    sv_filenames = np.array([['fpn', 'fpb', 'fpbh', 'fpby'],['tnn', 'tnb', 'tnbh', 'tnby'], \
                             ['fdn', 'fdb', 'fdbh', 'fdby'],['fnn', 'fnb', 'fnbh', 'fnby'], \
                             ['sfdn', 'sfdb', 'sfdbh', 'sfdby'], ['stpn', 'stpb', 'stpbh', 'stpby'], \
                             ['stnn', 'stnb', 'stnbh', 'stnby'], ['sfnn', 'sfnb', 'sfnbh', 'sfnby'], \
                             ['sfpn', 'sfpdb', 'sfpbh', 'sfpby'], ['tpn', 'tpb', 'tpbh', 'tpby'] ])    
    ##save the effect sizes and sample sizes
    file_handle = file('papy_output/effect_n_sample_sizes.txt', 'a')
    np.savetxt(file_handle, np.array(['effect sizes']), fmt='%s')
    np.savetxt(file_handle, effectSizes, delimiter="," , fmt='%.3f')
    np.savetxt(file_handle, np.array(['sample sizes']), fmt='%s')
    np.savetxt(file_handle, sampleSizes, delimiter=",", fmt='%.3f')
    file_handle.close()
        
    ##save files. jj- Metric options; kk- Correction options; ii- Variable number; for example: jj=1, kk=1 mean tpn-- true positive no correction. 
    
    for jj in range(0, sv_filenames.shape[0]):
        for kk in range(0, sv_filenames.shape[1]):
            file_handle = file('papy_output/diffgroups-%s.csv'%(sv_filenames[jj][kk]), 'a')
            ##write the title line with columns "variables, Sample Sizes (Effect Sizes as columns), and Effect Sizes"
            title_str=np.append(np.array([['Variables','Effect Sizes (Sample Sizes in Columns)']]), sampleSizes.astype('str'), axis=1)
            np.savetxt(file_handle, title_str, delimiter=',', fmt='%s')
            ##write rest of output matrix
            for ii in range(0, num_cols): ##num_cols is from the test dataset, means number of variables
                np.savetxt(file_handle, np.insert(diffgroups[ii][kk][jj],[0], np.insert(effectSizes.T, [0], np.ones([effectSizes.shape[1],1])*(ii+1), axis=1),axis=1), delimiter=",", fmt='%.5f')
            file_handle.close()
            
            file_handle = file('papy_output/linearregression-%s.csv'%(sv_filenames[jj][kk]), 'a')
            ##write the title line with columns "variables, Sample Sizes (Effect Sizes as columns), and Effect Sizes"
            title_str=np.append(np.array([['Variables','Effect Sizes (Sample Sizes in Columns)']]), sampleSizes.astype('str'), axis=1)
            np.savetxt(file_handle, title_str, delimiter=',', fmt='%s')
            ##write rest of matrix    
            for ii in range(0, num_cols): ##num_cols is from the test dataset, means number of variables
                np.savetxt(file_handle, np.insert(linearregression[ii][kk][jj],[0], np.insert(effectSizes.T, [0], np.ones([effectSizes.shape[1],1])*(ii+1), axis=1),axis=1), delimiter=",", fmt='%.5f')
            file_handle.close()
    ##iSurfacePlot(diffgroups, 2, 4,2 , sampleSizes, effectSizes,numberreps)
    
    ##plot the surfaces of power rate acrossing the combination of effectSize and SampleSize (classfied)
    iSurfacePlotTPR(output_uncTP_ratio_median, 'papy_output/plot-power-rate-noCorrection-diffgroups.html',  'no correction', sampleSizes, effectSizes, numberreps)
    iSurfacePlotTPR(output_bonfTP_ratio_median, 'papy_output/plot-power-rate-bonfCorrection-diffgroups.html',  'Bonferroni correction', sampleSizes, effectSizes, numberreps)
    iSurfacePlotTPR(output_bhTP_ratio_median, 'papy_output/plot-power-rate-bhCorrection-diffgroups.html',  'Benjamini-Hochberg correction', sampleSizes, effectSizes, numberreps)
    iSurfacePlotTPR(output_byTP_ratio_median, 'papy_output/plot-power-rate-byCorrection-diffgroups.html',  'Benjamini-Yekutieli correction', sampleSizes, effectSizes, numberreps)
     
    ##plot the slice of surfaces power rate; x-axis is based on sample size (columns)
    ## 2nd row, mid row, and the 2nd last row
    slice_rows = np.array([1, int(floor(effectSizes.shape[1]/2)), effectSizes.shape[1]-2]) 
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_rows:
        Y_temp.append(output_uncTP_ratio_median[ll, :])
        Y_std_temp.append(output_uncTP_ratio_iqr[ll, :])
    iSlicesPlot(sampleSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-noCorrection-diffgroups.html', \
                            'Proportion of Variables with Power (True Positive)> 0.8', \
                            'Sample Size', 'tpn', 'Effect Size=', effectSizes[:,slice_rows])
    
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_rows:
        Y_temp.append(output_bonfTP_ratio_median[ll, :])
        Y_std_temp.append(output_bonfTP_ratio_iqr[ll, :])                        
    iSlicesPlot(sampleSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-bonfCorrection-diffgroups.html', \
                            'Proportion of Variables with Power (True Positive)> 0.8', \
                            'Sample Size', 'tpb', 'Effect Size=', effectSizes[:,slice_rows])
                            
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_rows:
        Y_temp.append(output_bhTP_ratio_median[ll, :])
        Y_std_temp.append(output_bhTP_ratio_iqr[ll, :])                        
    iSlicesPlot(sampleSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-bhCorrection-diffgroups.html', \
                            'Proportion of Variables with Power (True Positive)> 0.8', \
                            'Sample Size', 'tpbh', 'Effect Size=', effectSizes[:,slice_rows])
                            
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_rows:
        Y_temp.append(output_byTP_ratio_median[ll, :])
        Y_std_temp.append(output_byTP_ratio_iqr[ll, :])                        
    iSlicesPlot(sampleSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-byCorrection-diffgroups.html', \
                            'Proportion of Variables with Power (True Positive)> 0.8', \
                            'Sample Size', 'tpby', 'Effect Size=', effectSizes[:,slice_rows])
                            
    ##plot the slice of surfaces power rate; x-axis is based on effect size (rows)
    ## 2nd col, mid col, and the 2nd last col
    slice_cols = np.array([1, int(floor(sampleSizes.shape[1]/2)), sampleSizes.shape[1]-2])
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_cols:
        Y_temp.append(output_uncTP_ratio_median[:, ll])
        Y_std_temp.append(output_uncTP_ratio_iqr[:, ll])
    iSlicesPlot(effectSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-noCorrection-diffgroups-eff.html', \
                            'Proportion of Variables with Power (True Positive)> 0.8', \
                            'Effect Size', 'tpn', 'Sample Size=', sampleSizes[:,slice_cols])
                            
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_cols:
        Y_temp.append(output_bonfTP_ratio_median[:, ll])
        Y_std_temp.append(output_bonfTP_ratio_iqr[:, ll])                        
    iSlicesPlot(effectSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-bonfCorrection-diffgroups-eff.html', \
                            'Proportion of Variables with Power (True Positive)> 0.8', \
                            'Effect Size', 'tpb', 'Sample Size=', sampleSizes[:,slice_cols])
                            
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_cols:
        Y_temp.append(output_bhTP_ratio_median[:, ll])
        Y_std_temp.append(output_bhTP_ratio_iqr[:, ll])                        
    iSlicesPlot(effectSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-bhCorrection-diffgroups-eff.html', \
                            'Proportion of Variables with Power (True Positive)> 0.8', \
                            'Effect Size', 'tpbh', 'Sample Size=', sampleSizes[:,slice_cols])
                            
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_cols:
        Y_temp.append(output_byTP_ratio_median[:, ll])
        Y_std_temp.append(output_byTP_ratio_iqr[:, ll])                        
    iSlicesPlot(effectSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-byCorrection-diffgroups-eff.html', \
                            'Proportion of Variables with Power (True Positive)> 0.8', \
                            'Effect Size', 'tpby', 'Sample Size=', sampleSizes[:,slice_cols])
    
    ##plot the surfaces of power rate acrossing the combination of effectSize and SampleSize (linear regression)
    iSurfacePlotTPR(output_uncTP_ratio_median_ln, 'papy_output/plot-power-rate-noCorrection-linearregression.html',  'no correction', sampleSizes, effectSizes, numberreps)
    iSurfacePlotTPR(output_bonfTP_ratio_median_ln, 'papy_output/plot-power-rate-bonfCorrection-linearregression.html',  'Bonferroni correction', sampleSizes, effectSizes, numberreps)
    iSurfacePlotTPR(output_bhTP_ratio_median_ln, 'papy_output/plot-power-rate-bhCorrection-linearregression.html',  'Benjamini-Hochberg correction', sampleSizes, effectSizes, numberreps)
    iSurfacePlotTPR(output_byTP_ratio_median_ln, 'papy_output/plot-power-rate-byCorrection-linearregression.html',  'Benjamini-Yekutieli correction', sampleSizes, effectSizes, numberreps)

    ## (linear regression)
    ##plot the slice of surfaces power rate; x-axis is based on sample size (columns)
    ## 2nd row, mid row, and the 2nd last row
    slice_rows = np.array([1, int(floor(effectSizes.shape[1]/2)), effectSizes.shape[1]-2]) 
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_rows:
        Y_temp.append(output_uncTP_ratio_median_ln[ll, :])
        Y_std_temp.append(output_uncTP_ratio_iqr_ln[ll, :])
    iSlicesPlot(sampleSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-noCorrection-ln.html', \
                            'Proportion of Variables with Power (True Positive)> 0.8 (linear-regression)', \
                            'Sample Size', 'tpn', 'Effect Size=', effectSizes[:,slice_rows])
    
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_rows:
        Y_temp.append(output_bonfTP_ratio_median_ln[ll, :])
        Y_std_temp.append(output_bonfTP_ratio_iqr_ln[ll, :])                        
    iSlicesPlot(sampleSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-bonfCorrection-ln.html', \
                            'Proportion of Variables with Power (True Positive)> 0.8 (linear-regression)', \
                            'Sample Size', 'tpb', 'Effect Size=', effectSizes[:,slice_rows])
                            
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_rows:
        Y_temp.append(output_bhTP_ratio_median_ln[ll, :])
        Y_std_temp.append(output_bhTP_ratio_iqr_ln[ll, :])                        
    iSlicesPlot(sampleSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-bhCorrection-ln.html', \
                            'Proportion of Variables with Power (True Positive)> 0.8 (linear-regression)', \
                            'Sample Size', 'tpbh', 'Effect Size=', effectSizes[:,slice_rows])
                            
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_rows:
        Y_temp.append(output_byTP_ratio_median_ln[ll, :])
        Y_std_temp.append(output_byTP_ratio_iqr_ln[ll, :])                        
    iSlicesPlot(sampleSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-byCorrection-ln.html', \
                            'Proportion of Variables with Power (True Positive)> 0.8 (linear-regression)', \
                            'Sample Size', 'tpby', 'Effect Size=', effectSizes[:,slice_rows])
                            
    ##plot the slice of surfaces power rate; x-axis is based on effect size (rows)
    ## 2nd col, mid col, and the 2nd last col
    slice_cols = np.array([1, int(floor(sampleSizes.shape[1]/2)), sampleSizes.shape[1]-2])
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_cols:
        Y_temp.append(output_uncTP_ratio_median_ln[:, ll])
        Y_std_temp.append(output_uncTP_ratio_iqr_ln[:, ll])
    iSlicesPlot(effectSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-noCorrection-ln-eff.html', \
                            'Proportion of Variables with Power (True Positive)> 0.8 (linear-regression)', \
                            'Effect Size', 'tpn', 'Sample Size=', sampleSizes[:,slice_cols])
                            
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_cols:
        Y_temp.append(output_bonfTP_ratio_median_ln[:, ll])
        Y_std_temp.append(output_bonfTP_ratio_iqr_ln[:, ll])                        
    iSlicesPlot(effectSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-bonfCorrection-ln-eff.html', \
                            'Proportion of Variables with Power (True Positive)> 0.8 (linear-regression)', \
                            'Effect Size', 'tpb', 'Sample Size=', sampleSizes[:,slice_cols])
                            
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_cols:
        Y_temp.append(output_bhTP_ratio_median_ln[:, ll])
        Y_std_temp.append(output_bhTP_ratio_iqr_ln[:, ll])                        
    iSlicesPlot(effectSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-bhCorrection-ln-eff.html', \
                            'Proportion of Variables with Power (True Positive)> 0.8 (linear-regression)', \
                            'Effect Size', 'tpbh', 'Sample Size=', sampleSizes[:,slice_cols])
                            
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_cols:
        Y_temp.append(output_byTP_ratio_median_ln[:, ll])
        Y_std_temp.append(output_byTP_ratio_iqr_ln[:, ll])                        
    iSlicesPlot(effectSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-byCorrection-ln-eff.html', \
                            'Proportion of Variables with Power (True Positive)> 0.8 (linear-regression)', \
                            'Effect Size', 'tpby', 'Sample Size=', sampleSizes[:,slice_cols])
    ##plot all surfac of True Positive
    ## sv_name1=np.array([['tpn', 'tpb', 'tpbh', 'tpby']])
    ## for ii in range(0, num_cols):
        ## for jj in range(0, sv_name1.shape[1]):
        ## ##print ii, jj, kk         
            ## iSurfacePlot(diffgroups, 'papy_output/plot-variable%d-diffgroups-%s.html'%(ii+1,sv_name1[0][jj]), ii+1,  10, jj+1,sampleSizes, effectSizes,numberreps)
    
    
    ##save and plot surface of mean of each variable; 
    ##sv_filenames.shape[0] is the dimension of metric options; 
    ##sv_filenames.shape[1] is the dimension of correction options
    for jj in range(0, sv_filenames.shape[0]):
        for kk in range(0, sv_filenames.shape[1]):
            temp_diffgroups_array=[]
            temp_linearregression_array=[]
            mean_diffgroups_array=[]
            mean_linearregression_array=[]    
            for ii in range(0, num_cols):
                temp_diffgroups_array.append(diffgroups[ii][kk][jj])
                temp_linearregression_array.append(linearregression[ii][kk][jj])
            temp_diffgroups_array=np.array(temp_diffgroups_array)
            temp_linearregression_array=np.array(temp_linearregression_array)
            
            mean_diffgroups_array=np.mean(temp_diffgroups_array,axis=0)
            mean_linearregression_array=np.mean(temp_linearregression_array, axis=0)
            #for calculating standard deviation
            std_diffgroups_array=np.std(temp_diffgroups_array,axis=0)
            std_linearregression_array=np.std(temp_linearregression_array, axis=0)
            
            
            file_handle = file('papy_output/mean-diffgroups-%s.csv'%(sv_filenames[jj][kk]), 'a')
            np.savetxt(file_handle, mean_diffgroups_array, delimiter=",", fmt='%.10f')
            file_handle.close()
            file_handle = file('papy_output/mean-linearregression-%s.csv'%(sv_filenames[jj][kk]), 'a')
            np.savetxt(file_handle, mean_linearregression_array, delimiter=",", fmt='%.10f')
            file_handle.close()
            
            ##plotting surface plots
            for ii in range(0,3):
                mean_diffgroups_array=np.expand_dims(mean_diffgroups_array, axis=0)
                mean_linearregression_array=np.expand_dims(mean_linearregression_array, axis=0)
            iSurfacePlot(mean_diffgroups_array, 'papy_output/plot-mean-diffgroups-%s.html'%(sv_filenames[jj][kk]), 1, 1, 1, sampleSizes, effectSizes,numberreps)
            iSurfacePlot(mean_linearregression_array, 'papy_output/plot-mean-linearregression-%s.html'%(sv_filenames[jj][kk]), 1, 1, 1, sampleSizes, effectSizes,numberreps)
    
    ##copy plotSurface.py to papy_output folder
    ##for plotting interactive surface plots for the variables separately
    ##shutil.copy2('plotSurface.py','papy_output')
    ##create a zip file on the output folder
    shutil.make_archive('papy_output_zip', 'zip', 'papy_output')
    
    ##copy some files for user viewing in results folder
    if not os.path.exists('results'):
        os.makedirs('results')
    shutil.copy2('papy_output/plot-power-rate-byCorrection-diffgroups.html','results')
    shutil.copy2('papy_output/plot-slice-power-rate-byCorrection-diffgroups.html','results')
    shutil.copy2('papy_output/plot-slice-power-rate-byCorrection-diffgroups-eff.html','results')
    shutil.copy2('papy_output/plot-power-rate-noCorrection-diffgroups.html','results')
    shutil.copy2('papy_output/plot-slice-power-rate-noCorrection-diffgroups.html','results')
    shutil.copy2('papy_output/plot-slice-power-rate-noCorrection-diffgroups-eff.html','results')
    
    ##delete the papy_output folder
    shutil.rmtree('papy_output')
    
    ##display user information
    print('The output files are in the papy_output_zip.zip in the running directory')
    print('Please move the papy_output_zip.zip file to your work directory and unzipped it to view the output files.')
    print('a Python script, plotSurface.py, is included for plotting interactive surface plots for variables')
    print('for more details, please have a look the .zip file.')
                                                      
if __name__=="__main__":
    ##detect python version#
    ver = sys.version
    if not ('2.7' in ver):
        print('This tool currently only runs in Python 2.7. Please install Python 2.7')
        exit(0)
                    
    ##start to parse input arguments
    args = sys.argv
    ## for i in range(1, len(args)):
        ## print(args[i],type(args[i]),len(args[i]))
        
    if (len(args)<3):
        print('too few arguments')
        print('simple usage: python pa.py TutorialData.csv 8, TutorialData.csv is input test data set, can be replaced by \n \n \
              actual data set name, 8 means the first 8 variables, which can be a range, e.g., 8-16 \n \n \n \
              full usage: python pa.py TutorialData.csv 2-9 0:100:500 0.05:0.05:0.7 20 4 \n \n \
              0:100:500 means the range of sample sizes from 0 to 500 (not inclusive) with interval of 100 \n \n \
              0.05:0.05:0.7 means the range of effect sizes from 0.05 to 0.7 (not inclusive) with interval of 0.05 \n \n \
              20 is an integer number of repeats. \n \n \
              4 is an integer number as number of CPU cores to use. ')
        exit(0)

    if (len(args)>3):
        tmpStr=args[3].split(':')
        if len(tmpStr)<3:
            print('the 3rd parameter is for defining the range of sample size with interval\n \
                  for example, python pa.py TutorialData.csv 2-9 0:50:500')
            exit(0)
    else:
        args.append('0:100:501')
        
    if (len(args)>4):
        tmpStr=args[4].split(':')
        if len(tmpStr)<3:
            print('the 4th parameter is for defining the range of effect size with interval\n \
                  for example, python pa.py TutorialData.csv 2-9 10:50:500 0.05:0.05:0.8')
            exit(0)
    else:
        args.append('0.05:0.05:0.8')
    
    if (len(args)>5):
        print('')
    else:
        args.append('10') 
    
    if (len(args)>6):
        tmpInt=int(args[6])
        if type(tmpInt).__name__=='int':
            if multiprocessing.cpu_count()-1 <= 0:
                cores = 1
            else:
                cores = multiprocessing.cpu_count()
                if tmpInt>cores:
                    args[6]=str(cores)
                    print('You machine does not have enough cores as you request, \n \
                          the maximum number of cores - %i - will be used instead' %(cores))
        else:
            print('the 6th parameter is for defining the number of CPU cores to use\n \
                  for example, python pa.py TutorialData.csv 2-9 10:50:500 0.05:0.05:0.8 10 4')
            exit(0)
    else:
        if multiprocessing.cpu_count()-1 <= 0:
            cores = 1
        else:
            cores = multiprocessing.cpu_count()   
        args.append(str(cores)) 
        
                        
    ## print('len of args is %i'%(len(args[1:])))
    main(args[1],args[2],args[3],args[4],args[5],args[6])
