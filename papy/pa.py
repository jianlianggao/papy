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
import matplotlib.pyplot as plt
import shutil                                           #for creating zip files
from math import fabs,floor,log,exp
from datetime import datetime
from joblib import Parallel, delayed                    #for Parallel computing
from statsmodels import robust                          #for work out median absolute deviation
'''
# For 3d plots. This import is necessary to have 3D plotting below
#from mpl_toolkits.mplot3d import Axes3D
# for saving the plot to pdf file 
#from matplotlib.backends.backend_pdf import PdfPages
'''
##=======Beginning of interactive SurfacePlot============
def iSurfacePlot(output, svfilename, variable,metric,correction, sizeeff,samplsizes,nreps):
    import plotly as py
    import plotly.graph_objs as go
    MUtot = output[variable-1][correction-1][metric-1]
    NS, NSE = MUtot.shape
    #SIGMAtot = output[variable-1][correction-1][metric+5-1]
    #SIGMAlow=MUtot-1.96*SIGMAtot/np.sqrt(nreps)
    #SIGMAlow = np.array([[0 if x<0 else x for x in y] for y in SIGMAlow])
    
    ##plot
    #generate a 2D grid
    X, Y = np.meshgrid(sizeeff, samplsizes)
    
    #debug
    ## print svfilename
    ## if 'mean' in svfilename:
        ## print 'plot mean'
    if metric == 1:
        if not 'mean' in svfilename:
            zaxis_title = 'True Positive Rate'
        else:
            if 'tp' in svfilename:
                ## print 'plot tp mean'
                zaxis_title = 'True Positive Rate'
            if 'fp' in svfilename:
                ## print 'plot fp mean'
                zaxis_title = 'False Positive Rate'
            if 'tn' in svfilename:
                ## print 'plot tn mean'
                zaxis_title = 'True Negative Rate'
            if 'fn' in svfilename:
                ## print 'plot fn mean'
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
                ## titlefont=dict(
                    ## family='Courier New, monospace',
                ## )
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

##====== Beginning of surface plots for power rate only===============
def iSurfacePlotTPR(output, svfilename, correction, sizeeff,samplsizes,nreps):
    import plotly as py
    import plotly.graph_objs as go
    MUtot = output
    NS, NSE = MUtot.shape
        
    ##plot
    #generate a 2D grid
    X, Y = np.meshgrid(sizeeff, samplsizes)
    
    #define z axis title
    zaxis_title = 'Power Rate'
    
    camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=2, y=2, z=0.1)
            )    
    layout = go.Layout(
        title='Power Rate (True Positive)-%s, %d Repeats'%(correction, nreps),
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
                ## titlefont=dict(
                    ## family='Courier New, monospace',
                ## )
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
##====== End of surface plots for power rate only===============

    
''' 
## This Surface plot method is scrapped.
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
    save_filename = 'resutls_%s.pdf'%(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    pp = PdfPages(save_filename)
    
    #give the PdfPages object to savefig()
    plt.savefig(pp, format='pdf')
    pp.savefig()
    pp.close()
    
    plt.show()
##=======End of SurfacePlot=========================
'''

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
    
    #np.random.seed(10)                                  ##add random seed for testing purpose    
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
    
    #define an array for storing the results in each step of repeat for all variables
    #with all effect sizes and sample sizes; 
    output_allsteps_tmp=[]
    
    output2 = Parallel(n_jobs=cores)(delayed(f_multiproc1)(sampSizes, signThreshold, effectSizes, nRepeats, nSampSizes, nEffSizes, Samples_seg, correlationMat_seg,ii) for ii in range(cores))            #n_jobs=-1 means using all CPUs or any other number>1 to specify the number of CPUs to use
    ## for ii in range(numVars):       # non parallelized loop
        ## f_multiproc(ii)
    ##pass the results to output
    output = []
    output.append(output2[0][0])
    output.append(output2[0][1])
    
    ##pass Power TPR results to output variables
    output_uncTP = []
    output_bonfTP = []
    output_bhTP = []
    output_byTP = []
    
    output_uncTP = np.array(output2[0][2])
    output_bonfTP = np.array(output2[0][3])
    output_bhTP = np.array(output2[0][4])
    output_byTP = np.array(output2[0][5])
    
    for ii in range(1, cores):
        output.append(output2[ii][0])
        output.append(output2[ii][1])
        
        output_uncTP = np.append(output_uncTP, output2[ii][2],axis=2)
        output_bonfTP = np.append(output_bonfTP, output2[ii][3],axis=2)
        output_bhTP = np.append(output_bhTP, output2[ii][4],axis=2)
        output_byTP = np.append(output_byTP, output2[ii][5],axis=2)
    
    output = np.array(output)    
    ##for the mean proportion of number of variables achieve the power; and the std
    output_uncTP_ratio_median = np.zeros((nEffSizes, nSampSizes))
    output_bonfTP_ratio_median = np.zeros((nEffSizes, nSampSizes))
    output_bhTP_ratio_median = np.zeros((nEffSizes, nSampSizes))
    output_byTP_ratio_median = np.zeros((nEffSizes, nSampSizes))
    
    output_uncTP_ratio_mad = np.zeros((nEffSizes, nSampSizes))
    output_bonfTP_ratio_mad = np.zeros((nEffSizes, nSampSizes))
    output_bhTP_ratio_mad = np.zeros((nEffSizes, nSampSizes))
    output_byTP_ratio_mad = np.zeros((nEffSizes, nSampSizes))
    
    for currEff in range(0, nEffSizes):
        for currSamp in range(0, nSampSizes):
            tmp_median_array = np.zeros(nRepeats)
            for currStep in range(0, nRepeats):
                tmp_median_array[currStep] = (sum(1 for x in output_uncTP[currEff][currSamp][:,currStep] if x>0.8))/float(numVars)
            output_uncTP_ratio_median[currEff][currSamp] = np.median(tmp_median_array)
            output_uncTP_ratio_mad[currEff][currSamp] = robust.mad(tmp_median_array)
            
            tmp_median_array = np.zeros(nRepeats)
            for currStep in range(0, nRepeats):
                tmp_median_array[currStep] = (sum(1 for x in output_bonfTP[currEff][currSamp][:,currStep] if x>0.8))/float(numVars)
            output_bonfTP_ratio_median[currEff][currSamp] = np.median(tmp_median_array)
            output_bonfTP_ratio_mad[currEff][currSamp] = robust.mad(tmp_median_array)
            
            tmp_median_array = np.zeros(nRepeats)
            for currStep in range(0, nRepeats):
                tmp_median_array[currStep] = (sum(1 for x in output_bhTP[currEff][currSamp][:,currStep] if x>0.8))/float(numVars)
            output_bhTP_ratio_median[currEff][currSamp] = np.median(tmp_median_array)
            output_bhTP_ratio_mad[currEff][currSamp] = robust.mad(tmp_median_array)
            
            tmp_median_array = np.zeros(nRepeats)
            for currStep in range(0, nRepeats):
                tmp_median_array[currStep] = (sum(1 for x in output_byTP[currEff][currSamp][:,currStep] if x>0.8))/float(numVars)            
            output_byTP_ratio_median[currEff][currSamp] = np.median(tmp_median_array)
            output_byTP_ratio_mad[currEff][currSamp] = robust.mad(tmp_median_array)
          
    try:
        return output, output_uncTP_ratio_median, output_bonfTP_ratio_median, output_bhTP_ratio_median, output_byTP_ratio_median,\
                output_uncTP_ratio_mad, output_bonfTP_ratio_mad, output_bhTP_ratio_mad, output_byTP_ratio_mad
    except:
        print 'error occurs when returning output'

def f_multiproc1(sampSizes, signThreshold, effectSizes, nRepeats, nSampSizes, nEffSizes, Samples_seg, correlationMat_seg, currCore):
    
    ## global Samples_seg, correlationMat_seg, nEffSizes, nSampSizes, nRepeats, sampSizes, effectSizes, output2
    #re-check numVars
    numVars = Samples_seg[currCore].shape[1]
    
    #for storing all results in all repeated steps with all effect sizes and sample
    #sizes for Power (TP) in current samples_seg
    output_all_uncTP_tmp=np.zeros((nEffSizes, nSampSizes, numVars, nRepeats))
    output_all_bonfTP_tmp=np.zeros((nEffSizes, nSampSizes, numVars, nRepeats))
    output_all_bhTP_tmp=np.zeros((nEffSizes, nSampSizes, numVars, nRepeats))
    output_all_byTP_tmp=np.zeros((nEffSizes, nSampSizes, numVars, nRepeats))
    
    #for storing results of all metric and correction options under the combination
    #of effect size and sample size for all variables
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
                    #np.random.seed(10)                                  ##add random seed for testing purpose
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
                        
                    #storing each result
                    output_all_uncTP_tmp[currEff][currSampSize][currVar][currRepeat]=uncTPTot
                    output_all_bonfTP_tmp[currEff][currSampSize][currVar][currRepeat]=bonfTPTot
                    output_all_bhTP_tmp[currEff][currSampSize][currVar][currRepeat]=bhTPTot
                    output_all_byTP_tmp[currEff][currSampSize][currVar][currRepeat]=byTPTot
                    
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
                storeVar[3][i] = byStruct[stats[i]]
                
        output.append(storeVar)
        ## storeVar1 = np.expand_dims(storeVar, axis=0)
        ## try:
            ## output=np.append(output,storeVar1,axis=0)
        ## except ValueError:
            ## output=storeVar1
        ## print output.shape
                  
    #output2[currVar].append(output)        
    print '|| \n'
    output.append(output_all_uncTP_tmp)
    output.append(output_all_bonfTP_tmp)
    output.append(output_all_bhTP_tmp)
    output.append(output_all_byTP_tmp)
    try:
        return output
    except:
        print 'error occurs when returning output in parallel'
    
def randperm1(totalLen):
    #function of random permuation and pick up the sub array according to the specified size
    
    #np.random.seed(10)                                  ##add random seed for testing purpose
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
    ## cores = 1
    print cores    
    Samples_seg = _chunkMatrix(Samples, cores)
    correlationMat_seg = _chunkMatrix(correlationMat, cores)
    
    output2=[]
    
    #define an array for storing the results in each step of repeat for all variables
    #with all effect sizes and sample sizes; 
    output_allsteps_tmp=[]
    
    output2 = Parallel(n_jobs=cores)(delayed(f_multiproc)(sampSizes, signThreshold, effectSizes, numVars, nRepeats, nSampSizes, nEffSizes, Samples_seg, correlationMat_seg, ii) for ii in range(cores))            #n_jobs=-1 means using all CPUs or any other number>1 to specify the number of CPUs to use
    ## for ii in range(numVars):
            ## f_multiproc(ii)
    ##pass the results to output
    output = []
    output.append(output2[0][0])
    output.append(output2[0][1])
    
    ##pass Power TPR results to output variables
    output_uncTP = []
    output_bonfTP = []
    output_bhTP = []
    output_byTP = []
    
    output_uncTP = np.array(output2[0][2])
    output_bonfTP = np.array(output2[0][3])
    output_bhTP = np.array(output2[0][4])
    output_byTP = np.array(output2[0][5])
    
    for ii in range(1, cores):
        output.append(output2[ii][0])
        output.append(output2[ii][1])
        
        output_uncTP = np.append(output_uncTP, output2[ii][2],axis=2)
        output_bonfTP = np.append(output_bonfTP, output2[ii][3],axis=2)
        output_bhTP = np.append(output_bhTP, output2[ii][4],axis=2)
        output_byTP = np.append(output_byTP, output2[ii][5],axis=2)
    
    output = np.array(output)    
    ##for the mean proportion of number of variables achieve the power; and the std
    output_uncTP_ratio_median = np.zeros((nEffSizes, nSampSizes))
    output_bonfTP_ratio_median = np.zeros((nEffSizes, nSampSizes))
    output_bhTP_ratio_median = np.zeros((nEffSizes, nSampSizes))
    output_byTP_ratio_median = np.zeros((nEffSizes, nSampSizes))
    
    output_uncTP_ratio_mad = np.zeros((nEffSizes, nSampSizes))
    output_bonfTP_ratio_mad = np.zeros((nEffSizes, nSampSizes))
    output_bhTP_ratio_mad = np.zeros((nEffSizes, nSampSizes))
    output_byTP_ratio_mad = np.zeros((nEffSizes, nSampSizes))
    
    for currEff in range(0, nEffSizes):
        for currSamp in range(0, nSampSizes):
            tmp_median_array = np.zeros(nRepeats)
            for currStep in range(0, nRepeats):
                tmp_median_array[currStep] = (sum(1 for x in output_uncTP[currEff][currSamp][:,currStep] if x>0.8))/float(numVars)
            output_uncTP_ratio_median[currEff][currSamp] = np.median(tmp_median_array)
            output_uncTP_ratio_mad[currEff][currSamp] = robust.mad(tmp_median_array)
            
            tmp_median_array = np.zeros(nRepeats)
            for currStep in range(0, nRepeats):
                tmp_median_array[currStep] = (sum(1 for x in output_bonfTP[currEff][currSamp][:,currStep] if x>0.8))/float(numVars)
            output_bonfTP_ratio_median[currEff][currSamp] = np.median(tmp_median_array)
            output_bonfTP_ratio_mad[currEff][currSamp] = robust.mad(tmp_median_array)
            
            tmp_median_array = np.zeros(nRepeats)
            for currStep in range(0, nRepeats):
                tmp_median_array[currStep] = (sum(1 for x in output_bhTP[currEff][currSamp][:,currStep] if x>0.8))/float(numVars)
            output_bhTP_ratio_median[currEff][currSamp] = np.median(tmp_median_array)
            output_bhTP_ratio_mad[currEff][currSamp] = robust.mad(tmp_median_array)
            
            tmp_median_array = np.zeros(nRepeats)
            for currStep in range(0, nRepeats):
                tmp_median_array[currStep] = (sum(1 for x in output_byTP[currEff][currSamp][:,currStep] if x>0.8))/float(numVars)            
            output_byTP_ratio_median[currEff][currSamp] = np.median(tmp_median_array)
            output_byTP_ratio_mad[currEff][currSamp] = robust.mad(tmp_median_array)            
            
    try:
        return output, output_uncTP_ratio_median, output_bonfTP_ratio_median, output_bhTP_ratio_median, output_byTP_ratio_median,\
                output_uncTP_ratio_mad, output_bonfTP_ratio_mad, output_bhTP_ratio_mad, output_byTP_ratio_mad
            
    except:
        print 'error occurs when returning output'
        
def f_multiproc(sampSizes, signThreshold, effectSizes, numVars, nRepeats, nSampSizes, nEffSizes, Samples_seg, correlationMat_seg, currCore):
    ## global Samples_seg, correlationMat_seg, output2, nEffSizes, nSampSizes, nRepeats, sampSizes, effectSizes
    ## global output2 # this definition doesn't work

    #re-check numVars
    numVars = Samples_seg[currCore].shape[1]
    
    #for storing all results in all repeated steps with all effect sizes and sample
    #sizes for Power (TP) in current samples_seg
    output_all_uncTP_tmp=np.zeros((nEffSizes, nSampSizes, numVars, nRepeats))
    output_all_bonfTP_tmp=np.zeros((nEffSizes, nSampSizes, numVars, nRepeats))
    output_all_bhTP_tmp=np.zeros((nEffSizes, nSampSizes, numVars, nRepeats))
    output_all_byTP_tmp=np.zeros((nEffSizes, nSampSizes, numVars, nRepeats))
    
    #output = {'EACHSTEP':np.zeros((numVars, nEffSizes, nSampSizes, nRepeats)), 'ALL':[]}
    output=[]
    
    if (nEffSizes == 1 and nSampSizes == 1):
        storeVar = np.zeros((4,nRepeats))
    elif (nEffSizes > 1 or nSampSizes > 1):
        storeVar = np.zeros((4,nRepeats, nEffSizes, nSampSizes))                            
    #define uncStruct -- structual data
    #STP-- State for True Positive prediction; SFP -- State for False Positive prediction
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
                #for debugging
                #output multiplerepeats results
                #file_handle = file('multiplerepeats-bonf.csv', 'a')
                    
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
                        
                    #storing each result
                    #output['EACHSTEP'][currVar][currEff][currSampSize][currRepeat]=bonfFNTot
                    output_all_uncTP_tmp[currEff][currSampSize][currVar][currRepeat]=uncTPTot
                    output_all_bonfTP_tmp[currEff][currSampSize][currVar][currRepeat]=bonfTPTot
                    output_all_bhTP_tmp[currEff][currSampSize][currVar][currRepeat]=bhTPTot
                    output_all_byTP_tmp[currEff][currSampSize][currVar][currRepeat]=byTPTot    
                #for debugging
                #output multiplerepeats results
                #np.savetxt(file_handle,  multiplerepeats.noCorrection['FN']) 
                #print currVar, currSampSize, currEff
                #print multiplerepeats.Bonferroni['TN']   
                #for debugging
                #output multiplerepeats results
                #file_handle.close()        
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
                storeVar[3][i] = byStruct[stats[i]]
            
        output.append(storeVar)
    print '| \n'
    output.append(output_all_uncTP_tmp)
    output.append(output_all_bonfTP_tmp)
    output.append(output_all_bhTP_tmp)
    output.append(output_all_byTP_tmp)
    try:        
        return output
    except:        
        print 'error occurs when returning output in parallel'         
    
    
    
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
    #function of random permuation and pick up the sub array according to the specified size
    #np.random.seed(10)                                  ##add random seed for testing purpose
    tempList = np.random.permutation(totalLen)                  ##generate a random permutation array
    #random.seed(10)                                  ##add random seed for testing purpose
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
    try: #TPR - power
        TPtot = TP/(TP+FN)
    except ZeroDivisionError:
        #TPtot = float('NaN')
        TPtot = 0.0
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
def main(argv1, argv2): 
    ##take input arguments
    print argv1
    print argv2   
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
    
    print 'Input data matrix size is :' + str(rows) + ',' + str(cols)

    effectSizes = np.array([[0.05, 0.1, 0.15,0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]])
    sampleSizes = np.array([[1, 50, 100, 200, 250, 350, 500, 750, 1000]])
    
    #define output metric options
    metric_opt = np.array([1, 2, 3, 4])  #see options description below
    correction_opt = np.array([1, 2, 3, 4]) #see correction options description below
    
    
    numberreps= 10
    ## ## Calculat for a subset of 4 variables (less than 20 seconds on 4-core desktop for each analysis)
    diffgroups = np.array([])
    linearregression = np.array([])
    t_start = datetime.now()
    num_cols = int(argv2)
    if (num_cols > 0):
        diffgroups, output_uncTP_ratio_median, output_bonfTP_ratio_median, output_bhTP_ratio_median, output_byTP_ratio_median,\
                output_uncTP_ratio_mad, output_bonfTP_ratio_mad, output_bhTP_ratio_mad, output_byTP_ratio_mad \
                = PCalc_2Group(XSRV[:,np.arange(0,num_cols)],effectSizes, sampleSizes, 0.05, 5000, numberreps)
        linearregression, output_uncTP_ratio_median_ln, output_bonfTP_ratio_median_ln, output_bhTP_ratio_median_ln, output_byTP_ratio_median_ln,\
                output_uncTP_ratio_mad_ln, output_bonfTP_ratio_mad_ln, output_bhTP_ratio_mad_ln, output_byTP_ratio_mad_ln \
                 = PCalc_Continuous(XSRV[:,np.arange(0,num_cols)],effectSizes, sampleSizes, 0.05, 5000, numberreps)
        t_end = datetime.now()
        print 'Time collapsed: ' + str(t_end-t_start)
   
    else:
        t_start = datetime.now()
        diffgroups, output_uncTP_ratio_median, output_bonfTP_ratio_median, output_bhTP_ratio_median, output_byTP_ratio_median,\
                output_uncTP_ratio_mad, output_bonfTP_ratio_mad, output_bhTP_ratio_mad, output_byTP_ratio_mad \
                = PCalc_2Group(XSRV,effectSizes, sampleSizes, 0.05, 5000, numberreps)
        linearregression, output_uncTP_ratio_median_ln, output_bonfTP_ratio_median_ln, output_bhTP_ratio_median_ln, output_byTP_ratio_median_ln,\
                output_uncTP_ratio_mad_ln, output_bonfTP_ratio_mad_ln, output_bhTP_ratio_mad_ln, output_byTP_ratio_mad_ln \
                = PCalc_Continuous(XSRV,effectSizes, sampleSizes, 0.05, 5000, numberreps)
        t_end = datetime.now()
        print 'Time collapsed: ' + str(t_end-t_start)

    #diffgroups has dimension of (number of variables, 4, 10, effectsize, samplesize);
    #number of variables is the input number of columns from the input dataset.
    #4-- 4 correction options
    #10--10 metric as "TP","FP","TN","FN","FD","STP","SFP","STN","SFN","SFD" 

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
    #write diffgroups and linearregression into file for testing purpose
    #np.savetxt('diffgroups.csv',diffgroups[1][3][1], delimiter=",")
    #np.savetxt('linearregression.csv',linearregression[1][3][1], delimiter=",")
    if not os.path.exists('papy_output'):
        os.makedirs('papy_output')
    
    #file names matrix
    sv_filenames = np.array([['tpn', 'tpb', 'tpbh', 'tpby'],['fpn', 'fpb', 'fpbh', 'fpby'], \
                             ['tnn', 'tnb', 'tnbh', 'tnby'],['fnn', 'fnb', 'fnbh', 'fnby']])    
    #save the effect sizes and sample sizes
    file_handle = file('papy_output/effect_n_sample_sizes.txt', 'a')
    np.savetxt(file_handle, np.array(['effect sizes']), fmt='%s')
    np.savetxt(file_handle, effectSizes, delimiter="," , fmt='%.3f')
    np.savetxt(file_handle, np.array(['sample sizes']), fmt='%s')
    np.savetxt(file_handle, sampleSizes, delimiter=",", fmt='%.3f')
    file_handle.close()
        
    #save files. jj- Metric options; kk- Correction options; ii- Variable number; for example: jj=1, kk=1 mean tpn-- true positive no correction. 
    for jj in range(0, sv_filenames.shape[0]):
        for kk in range(0, sv_filenames.shape[1]):
            file_handle = file('papy_output/diffgroups-%s.csv'%(sv_filenames[jj][kk]), 'a')
            for ii in range(0, num_cols):            
                np.savetxt(file_handle, np.array(['variable %s'%(str(ii+1))]), fmt='%s')
                np.savetxt(file_handle, diffgroups[ii][jj][kk], delimiter=",", fmt='%.5f')
            file_handle.close()
    
            file_handle = file('papy_output/linearregression-%s.csv'%(sv_filenames[jj][kk]), 'a')
            for ii in range(0, num_cols):            
                np.savetxt(file_handle, np.array(['variable %s'%(str(ii+1))]), fmt='%s')
                np.savetxt(file_handle, linearregression[ii][jj][kk], delimiter=",", fmt='%.5f')
            file_handle.close()
    #iSurfacePlot(diffgroups, 2, 4,2 , sampleSizes, effectSizes,numberreps)
    
    #plot the surfaces of power rate acrossing the combination of effectSize and SampleSize (classfied)
    iSurfacePlotTPR(output_uncTP_ratio_median, 'papy_output/plot-power-rate-noCorrection-diffgroups.html',  'no correction', sampleSizes, effectSizes, numberreps)
    iSurfacePlotTPR(output_bonfTP_ratio_median, 'papy_output/plot-power-rate-bonfCorrection-diffgroups.html',  'Bonferroni correction', sampleSizes, effectSizes, numberreps)
    iSurfacePlotTPR(output_bhTP_ratio_median, 'papy_output/plot-power-rate-bhCorrection-diffgroups.html',  'Benjamini-Hochberg correction', sampleSizes, effectSizes, numberreps)
    iSurfacePlotTPR(output_byTP_ratio_median, 'papy_output/plot-power-rate-byCorrection-diffgroups.html',  'Benjamini-Yekutieli correction', sampleSizes, effectSizes, numberreps)
    
    #plot the slice of surfaces power rate; x-axis is based on sample size (columns)
    # 2nd row, mid row, and the 2nd last row
    slice_rows = np.array([1, int(floor(effectSizes.shape[1]/2)), effectSizes.shape[1]-2]) 
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_rows:
        Y_temp.append(output_uncTP_ratio_median[ll, :])
        Y_std_temp.append(output_uncTP_ratio_mad[ll, :])
    iSlicesPlot(sampleSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-noCorrection-diffgroups.html', \
                            'plot-slice-power-rate-diffgroups', \
                            'Sample Size', 'tpn', 'Effect Size=', effectSizes[:,slice_rows])
    
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_rows:
        Y_temp.append(output_bonfTP_ratio_median[ll, :])
        Y_std_temp.append(output_bonfTP_ratio_mad[ll, :])                        
    iSlicesPlot(sampleSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-bonfCorrection-diffgroups.html', \
                            'plot-slice-power-rate-diffgroups', \
                            'Sample Size', 'tpb', 'Effect Size=', effectSizes[:,slice_rows])
                            
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_rows:
        Y_temp.append(output_bhTP_ratio_median[ll, :])
        Y_std_temp.append(output_bhTP_ratio_mad[ll, :])                        
    iSlicesPlot(sampleSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-bhCorrection-diffgroups.html', \
                            'plot-slice-power-rate-diffgroups', \
                            'Sample Size', 'tpbh', 'Effect Size=', effectSizes[:,slice_rows])
                            
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_rows:
        Y_temp.append(output_byTP_ratio_median[ll, :])
        Y_std_temp.append(output_byTP_ratio_mad[ll, :])                        
    iSlicesPlot(sampleSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-byCorrection-diffgroups.html', \
                            'plot-slice-power-rate-diffgroups', \
                            'Sample Size', 'tpby', 'Effect Size=', effectSizes[:,slice_rows])
                            
    #plot the slice of surfaces power rate; x-axis is based on effect size (rows)
    # 2nd col, mid col, and the 2nd last col
    slice_cols = np.array([1, int(floor(sampleSizes.shape[1]/2)), sampleSizes.shape[1]-2])
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_cols:
        Y_temp.append(output_uncTP_ratio_median[:, ll])
        Y_std_temp.append(output_uncTP_ratio_mad[:, ll])
    iSlicesPlot(effectSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-noCorrection-diffgroups-eff.html', \
                            'plot-slice-power-rate-diffgroups', \
                            'Effect Size', 'tpn', 'Sample Size=', sampleSizes[:,slice_cols])
                            
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_cols:
        Y_temp.append(output_bonfTP_ratio_median[:, ll])
        Y_std_temp.append(output_bonfTP_ratio_mad[:, ll])                        
    iSlicesPlot(effectSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-bonfCorrection-diffgroups-eff.html', \
                            'plot-slice-power-rate-diffgroups', \
                            'Effect Size', 'tpb', 'Sample Size=', sampleSizes[:,slice_cols])
                            
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_cols:
        Y_temp.append(output_bhTP_ratio_median[:, ll])
        Y_std_temp.append(output_bhTP_ratio_mad[:, ll])                        
    iSlicesPlot(effectSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-bhCorrection-diffgroups-eff.html', \
                            'plot-slice-power-rate-diffgroups', \
                            'Effect Size', 'tpbh', 'Sample Size=', sampleSizes[:,slice_cols])
                            
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_cols:
        Y_temp.append(output_byTP_ratio_median[:, ll])
        Y_std_temp.append(output_byTP_ratio_mad[:, ll])                        
    iSlicesPlot(effectSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-byCorrection-diffgroups-eff.html', \
                            'plot-slice-power-rate-diffgroups', \
                            'Effect Size', 'tpby', 'Sample Size=', sampleSizes[:,slice_cols])
    
    #plot the surfaces of power rate acrossing the combination of effectSize and SampleSize (linear regression)
    iSurfacePlotTPR(output_uncTP_ratio_median_ln, 'papy_output/plot-power-rate-noCorrection-linearregression.html',  'no correction', sampleSizes, effectSizes, numberreps)
    iSurfacePlotTPR(output_bonfTP_ratio_median_ln, 'papy_output/plot-power-rate-bonfCorrection-linearregression.html',  'Bonferroni correction', sampleSizes, effectSizes, numberreps)
    iSurfacePlotTPR(output_bhTP_ratio_median_ln, 'papy_output/plot-power-rate-bhCorrection-linearregression.html',  'Benjamini-Hochberg correction', sampleSizes, effectSizes, numberreps)
    iSurfacePlotTPR(output_byTP_ratio_median_ln, 'papy_output/plot-power-rate-byCorrection-linearregression.html',  'Benjamini-Yekutieli correction', sampleSizes, effectSizes, numberreps)

    # (linear regression)
    #plot the slice of surfaces power rate; x-axis is based on sample size (columns)
    # 2nd row, mid row, and the 2nd last row
    slice_rows = np.array([1, int(floor(effectSizes.shape[1]/2)), effectSizes.shape[1]-2]) 
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_rows:
        Y_temp.append(output_uncTP_ratio_median_ln[ll, :])
        Y_std_temp.append(output_uncTP_ratio_mad_ln[ll, :])
    iSlicesPlot(sampleSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-noCorrection-ln.html', \
                            'slice-power-rate-linear-regression', \
                            'Sample Size', 'tpn', 'Effect Size=', effectSizes[:,slice_rows])
    
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_rows:
        Y_temp.append(output_bonfTP_ratio_median_ln[ll, :])
        Y_std_temp.append(output_bonfTP_ratio_mad_ln[ll, :])                        
    iSlicesPlot(sampleSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-bonfCorrection-ln.html', \
                            'slice-power-rate-linear-regression', \
                            'Sample Size', 'tpb', 'Effect Size=', effectSizes[:,slice_rows])
                            
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_rows:
        Y_temp.append(output_bhTP_ratio_median_ln[ll, :])
        Y_std_temp.append(output_bhTP_ratio_mad_ln[ll, :])                        
    iSlicesPlot(sampleSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-bhCorrection-ln.html', \
                            'slice-power-rate-linear-regression', \
                            'Sample Size', 'tpbh', 'Effect Size=', effectSizes[:,slice_rows])
                            
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_rows:
        Y_temp.append(output_byTP_ratio_median_ln[ll, :])
        Y_std_temp.append(output_byTP_ratio_mad_ln[ll, :])                        
    iSlicesPlot(sampleSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-byCorrection-ln.html', \
                            'slice-power-rate-linear-regression', \
                            'Sample Size', 'tpby', 'Effect Size=', effectSizes[:,slice_rows])
                            
    #plot the slice of surfaces power rate; x-axis is based on effect size (rows)
    # 2nd col, mid col, and the 2nd last col
    slice_cols = np.array([1, int(floor(sampleSizes.shape[1]/2)), sampleSizes.shape[1]-2])
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_cols:
        Y_temp.append(output_uncTP_ratio_median_ln[:, ll])
        Y_std_temp.append(output_uncTP_ratio_mad_ln[:, ll])
    iSlicesPlot(effectSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-noCorrection-ln-eff.html', \
                            'slice-power-rate-linear-regression', \
                            'Effect Size', 'tpn', 'Sample Size=', sampleSizes[:,slice_cols])
                            
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_cols:
        Y_temp.append(output_bonfTP_ratio_median_ln[:, ll])
        Y_std_temp.append(output_bonfTP_ratio_mad_ln[:, ll])                        
    iSlicesPlot(effectSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-bonfCorrection-ln-eff.html', \
                            'slice-power-rate-linear-regression', \
                            'Effect Size', 'tpb', 'Sample Size=', sampleSizes[:,slice_cols])
                            
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_cols:
        Y_temp.append(output_bhTP_ratio_median_ln[:, ll])
        Y_std_temp.append(output_bhTP_ratio_mad_ln[:, ll])                        
    iSlicesPlot(effectSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-bhCorrection-ln-eff.html', \
                            'slice-power-rate-linear-regression', \
                            'Effect Size', 'tpbh', 'Sample Size=', sampleSizes[:,slice_cols])
                            
    Y_temp=[]
    Y_std_temp=[]
    for ll in slice_cols:
        Y_temp.append(output_byTP_ratio_median_ln[:, ll])
        Y_std_temp.append(output_byTP_ratio_mad_ln[:, ll])                        
    iSlicesPlot(effectSizes[0], Y_temp, Y_std_temp, \
                            'papy_output/plot-slice-power-rate-byCorrection-ln-eff.html', \
                            'slice-power-rate-linear-regression', \
                            'Effect Size', 'tpby', 'Sample Size=', sampleSizes[:,slice_cols])
    
    
    
    #plot all surface.
    for jj in range(0, sv_filenames.shape[0]):
        for kk in range(0, sv_filenames.shape[1]):
            for ii in range(0, num_cols):
                #print ii, jj, kk         
                iSurfacePlot(diffgroups, 'papy_output/plot-variable%d-diffgroups-%s.html'%(ii+1,sv_filenames[jj][kk]), ii+1, jj+1, kk+1, sampleSizes, effectSizes,numberreps)
                iSurfacePlot(linearregression, 'papy_output/plot-variable%d-linearregression-%s.html'%(ii+1,sv_filenames[jj][kk]), ii+1, jj+1, kk+1, sampleSizes, effectSizes,numberreps)
                #plotting slices of variables surface plots based on sample size
                #for diffgroups
                ## Y_eff_diffgroups=[]
                ## Y_eff_std_diffgroups=[]
                ## for ll in range(0,len(effectSizes[0])):
                    ## Y_eff_diffgroups.append(diffgroups[ii][kk][jj][ll,:])
                    ## Y_eff_std_diffgroups.append([0])
                ## iSlicesPlot(sampleSizes[0], Y_eff_diffgroups, Y_eff_std_diffgroups, \
                            ## 'papy_output/plot-slice-samp-variable%d-diffgroups-%s.html'%(ii+1,sv_filenames[jj][kk]), \
                            ## 'plot-slice-variable%d-diffgroups-%s'%(ii+1,sv_filenames[jj][kk]), \
                            ## 'Sample Size', sv_filenames[jj][kk], 'Effect Size=', effectSizes)
                #for linearregression            
                ## Y_eff_linearregression=[]
                ## Y_eff_std_linearregression=[]
                ## for ll in range(0,len(effectSizes[0])):
                    ## Y_eff_linearregression.append(linearregression[ii][kk][jj][ll,:])
                    ## Y_eff_std_linearregression.append([0])
                ## iSlicesPlot(sampleSizes[0], Y_eff_linearregression, Y_eff_std_linearregression, \
                            ## 'papy_output/plot-slice-samp-variable%d-linearregression-%s.html'%(ii+1,sv_filenames[jj][kk]), \
                            ## 'plot-slice-variable%d-linearregression-%s'%(ii+1,sv_filenames[jj][kk]), \
                            ## 'Sample Size', sv_filenames[jj][kk], 'Effect Size=', effectSizes)
                            
                #plotting slices of variables surface plots based on effect size
                #for diffgroups
                ## Y_eff_diffgroups=[]
                ## Y_eff_std_diffgroups=[]
                ## for ll in range(0,len(sampleSizes[0])):
                    ## Y_eff_diffgroups.append(diffgroups[ii][kk][jj][:,ll])
                    ## Y_eff_std_diffgroups.append([0])
                ## iSlicesPlot(effectSizes[0], Y_eff_diffgroups, Y_eff_std_diffgroups, \
                            ## 'papy_output/plot-slice-eff-variable%d-diffgroups-%s.html'%(ii+1,sv_filenames[jj][kk]), \
                            ## 'plot-slice-variable%d-diffgroups-%s'%(ii+1,sv_filenames[jj][kk]), \
                            ## 'Effect Size', sv_filenames[jj][kk], 'Sample Size=', sampleSizes)
                #for linearregression            
                ## Y_eff_linearregression=[]
                ## Y_eff_std_linearregression=[]
                ## for ll in range(0,len(sampleSizes[0])):
                    ## Y_eff_linearregression.append(linearregression[ii][kk][jj][:,ll])
                    ## Y_eff_std_linearregression.append([0])
                ## iSlicesPlot(effectSizes[0], Y_eff_linearregression, Y_eff_std_linearregression, \
                            ## 'papy_output/plot-slice-eff-variable%d-linearregression-%s.html'%(ii+1,sv_filenames[jj][kk]), \
                            ## 'plot-slice-variable%d-linearregression-%s'%(ii+1,sv_filenames[jj][kk]), \
                            ## 'Effect Size', sv_filenames[jj][kk], 'Sample Size=', sampleSizes)

    
    
    #debug
    #print diffgroups.shape
    
    #save and plot surface of mean of each variable; 
    #sv_filenames.shape[0] is the dimension of metric options; 
    #sv_filenames.shape[1] is the dimension of correction options
    for jj in range(0, sv_filenames.shape[0]):
        for kk in range(0, sv_filenames.shape[1]):
            temp_diffgroups_array=[]
            temp_linearregression_array=[]
            mean_diffgroups_array=[]
            mean_linearregression_array=[]    
            for ii in range(0, num_cols):
                temp_diffgroups_array.append(diffgroups[ii][jj][kk])
                temp_linearregression_array.append(linearregression[ii][jj][kk])
            temp_diffgroups_array=np.array(temp_diffgroups_array)
            temp_linearregression_array=np.array(temp_linearregression_array)
            
            mean_diffgroups_array=np.mean(temp_diffgroups_array,axis=0)
            mean_linearregression_array=np.mean(temp_linearregression_array, axis=0)
            #for calculating standard deviation
            std_diffgroups_array=np.std(temp_diffgroups_array,axis=0)
            std_linearregression_array=np.std(temp_linearregression_array, axis=0)
            
            
            file_handle = file('papy_output/mean-diffgroups-%s.csv'%(sv_filenames[jj][kk]), 'a')
            np.savetxt(file_handle, mean_diffgroups_array, delimiter=",", fmt='%.5f')
            file_handle.close()
            file_handle = file('papy_output/mean-linearregression-%s.csv'%(sv_filenames[jj][kk]), 'a')
            np.savetxt(file_handle, mean_linearregression_array, delimiter=",", fmt='%.5f')
            file_handle.close()
            
            #plotting slices of mean surface plots based on sample size
            ## Y_eff_mean_diffgroups=[]
            ## Y_eff_std_diffgroups=[]
            ## for ii in range(0,len(effectSizes[0])):
                ## Y_eff_mean_diffgroups.append(mean_diffgroups_array[ii,:])
                ## Y_eff_std_diffgroups.append(std_diffgroups_array[ii,:])                                
            ## iSlicesPlot(sampleSizes[0], Y_eff_mean_diffgroups, Y_eff_std_diffgroups, \
                        ## 'papy_output/plot-slice-samp-mean-diffgroups-%s.html'%(sv_filenames[jj][kk]),\
                        ## 'plot-slice-mean-diffgroups-%s'%(sv_filenames[jj][kk]), \
                        ## 'Sample Size', sv_filenames[jj][kk], 'Effect Size=', effectSizes)
                        
            ## Y_eff_mean_linearregression=[]
            ## Y_eff_std_linearregression=[]
            ## for ii in range(0,len(effectSizes[0])):
                ## Y_eff_mean_linearregression.append(mean_linearregression_array[ii,:])
                ## Y_eff_std_linearregression.append(std_linearregression_array[ii,:])                                
            ## iSlicesPlot(sampleSizes[0], Y_eff_mean_linearregression, Y_eff_std_linearregression, \
                        ## 'papy_output/plot-slice-samp-mean-linearregression-%s.html'%(sv_filenames[jj][kk]),\
                        ## 'plot-slice-mean-linearregression-%s'%(sv_filenames[jj][kk]), \
                        ## 'Sample Size', sv_filenames[jj][kk], 'Effect Size=', effectSizes)
            
            #plotting slices of mean surface plots based on effect size
            ## Y_eff_mean_diffgroups=[]
            ## Y_eff_std_diffgroups=[]
            ## for ii in range(0,len(sampleSizes[0])):
                ## Y_eff_mean_diffgroups.append(mean_diffgroups_array[:,ii])
                ## Y_eff_std_diffgroups.append(std_diffgroups_array[:,ii])                                
            ## iSlicesPlot(effectSizes[0], Y_eff_mean_diffgroups, Y_eff_std_diffgroups, \
                        ## 'papy_output/plot-slice-eff-mean-diffgroups-%s.html'%(sv_filenames[jj][kk]),\
                        ## 'plot-slice-mean-diffgroups-%s'%(sv_filenames[jj][kk]), \
                        ## 'Effect Size', sv_filenames[jj][kk], 'Sample Size=', sampleSizes)
                        
            ## Y_eff_mean_linearregression=[]
            ## Y_eff_std_linearregression=[]
            ## for ii in range(0,len(sampleSizes[0])):
                ## Y_eff_mean_linearregression.append(mean_linearregression_array[:,ii])
                ## Y_eff_std_linearregression.append(std_linearregression_array[:,ii])                                
            ## iSlicesPlot(effectSizes[0], Y_eff_mean_linearregression, Y_eff_std_linearregression, \
                        ## 'papy_output/plot-slice-eff-mean-linearregression-%s.html'%(sv_filenames[jj][kk]),\
                        ## 'plot-slice-mean-linearregression-%s'%(sv_filenames[jj][kk]), \
                        ## 'Effect Size', sv_filenames[jj][kk], 'Sample Size=', sampleSizes)
            
            #plotting surface plots
            for ii in range(0,3):
                mean_diffgroups_array=np.expand_dims(mean_diffgroups_array, axis=0)
                mean_linearregression_array=np.expand_dims(mean_linearregression_array, axis=0)
            iSurfacePlot(mean_diffgroups_array, 'papy_output/plot-mean-diffgroups-%s.html'%(sv_filenames[jj][kk]), 1, 1, 1, sampleSizes, effectSizes,numberreps)
            iSurfacePlot(mean_linearregression_array, 'papy_output/plot-mean-linearregression-%s.html'%(sv_filenames[jj][kk]), 1, 1, 1, sampleSizes, effectSizes,numberreps)
            
    #create a zip file on the output folder
    shutil.make_archive('papy_output_zip', 'zip', 'papy_output')
            
    #delete the papy_output folder
    shutil.rmtree('papy_output')
                                                      
if __name__=="__main__":
    #try:
        main(sys.argv[1], sys.argv[2])
    #except:
    #    print 'usage: python pa.py <data filename> <number of columns, 0 for use whole data set>'
