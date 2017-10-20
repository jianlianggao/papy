#!/usr/bin/env python

import os, sys, csv, inspect, dis, os.path, shutil
import numpy as np
import pandas as pd
from math import floor

def iSurfacePlot(output, svfilename, variable, metric, correction, samplsizes,sizeeff):
    """
    This is for plotting interactive 3D surface plot for mean of all variables.


    :param output: array for plotting.2D numpy array
    :type output: array
    :param svfilename: filename for saving the corresponding plot.
    :type svfilename: String
    :param variable: variable index nubmer. for plotting mean of all variables, as the size is 1, therefore use 1 as input parameter.
    :type variable: int
    :param metric: metric (confusion matrix, 'TP', 'TF' etc) index nubmer. for plotting mean of all variables, as the size is 1, therefore use 1 as input parameter.
    :type metric: int
    :param correction: correction methods (no correction, Bonferroni, Benjamini-Hochberg, Benjamini-Yekutieli index nubmer. for plotting mean of all variables, as the size is 1, therefore use 1 as input parameter.
    :type correction: int
    :param samplsizes: sample size matrix, numpy array 1 x n
    :type samplsizes: array
    :param sizeeff: effect size matrix, numpy array 1 x n
    :type sizeeff:  array
    :param nreps: number of repeats
    :type nreps: int
    :return:
    """
    import plotly as py
    import plotly.graph_objs as go
    MUtot = output[variable - 1][correction - 1][metric - 1]

    ##plot
    ##generate a 2D grid
    X, Y = np.meshgrid(samplsizes, sizeeff)

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
    camera = dict(
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
                range=[0, np.max(X) + 0.1]
            ),
            yaxis=dict(
                title='Effect Sizes',
                tickmode='linear',
                tick0=0,
                dtick=0.1,
                range=[0, np.max(Y)]
            ),
            zaxis=dict(
                title=zaxis_title,
                tickmode='linear',
                tick0=0,
                dtick=0.1,
                range=[0, 1.0]
            )
        )
    )
    data = [go.Surface(x=X, y=Y, z=MUtot)]
    fig = go.Figure(data=data, layout=layout)
    fig['layout'].update(scene=dict(camera=camera))
    py.offline.plot(fig, filename=svfilename, auto_open=False)


def iSurfacePlotTPR(output, svfilename, correction, samplsizes, sizeeff):
    import plotly as py
    import plotly.graph_objs as go
    MUtot = output
    NS, NSE = MUtot.shape

    ##plot
    ##generate a 2D grid
    X, Y = np.meshgrid(samplsizes, sizeeff)

    ##define z axis title
    zaxis_title = 'Proportion'

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=2, y=2, z=0.1)
    )
    layout = go.Layout(
        title='Proportion of Variables with Power (True Positive)> 0.8 -%s' % (correction),
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
                range=[0, np.max(X) + 0.1]
            ),
            yaxis=dict(
                title='Effect Sizes',
                tickmode='linear',
                tick0=0,
                dtick=0.1,
                range=[0, np.max(Y)]
            ),
            zaxis=dict(
                title=zaxis_title,
                tickmode='linear',
                tick0=0,
                dtick=0.1,
                range=[0, 1.0]
            )
        )
    )
    data = [go.Surface(x=X, y=Y, z=MUtot)]
    fig = go.Figure(data=data, layout=layout)
    fig['layout'].update(scene=dict(camera=camera))
    py.offline.plot(fig, filename=svfilename, auto_open=False)

def iSlicesPlot(X, Y, svfilename, plot_title, x_caption, y_caption, trace_label, trace_num):
    """
    For plotting slices from surface plots. Interactive plots with error bars.


    :param X: matrix for x-axis. either sample size or effect size. Use 1 x n numpy array.
    :type X: array
    :param Y: matrix of mean proportion of variables reach power>threshold.
    :type Y: array
    :param svfilename: filename for saving plots in the running folder.
    :type  svfilename: string
    :param plot_title: Title for the plot
    :type plot_title: String
    :param x_caption: Caption for x-axis
    :type x_caption: String
    :param y_caption: Caption for y-axis
    :type y_caption: String
    :param trace_label: for each trace to show relevant txt.
    :type trace_label: String
    :param trace_num: for trace label when mouse cursor moving along plot to show which line is which
    :type trace_num: array
    :return:
    """
    import plotly as py
    import plotly.graph_objs as go

    traces = []
    for ii in range(0, len(Y)):
        trace_tmp = go.Scatter(x=X, y=Y[ii],
                               name=trace_label + str(trace_num[0][ii])
                               )
        traces.append(trace_tmp)

    data = go.Data(traces)

    ##define other features of plots

    layout = go.Layout(
        title=plot_title,
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
    py.offline.plot(fig, filename=svfilename, auto_open=False)


def SubDirPath (d):
    return (filter(os.path.isdir, [os.path.join(d,f) for f in os.listdir(d)]))

def patternFilter(list1):
    import re
    newlist=[]
    for ll in list1:
        if re.search('[0-9]-[0-9]',ll):
            #tmp=filter(None, re.split("[\\ \-!?:]+", ll))
            tmp=ll.split(os.sep)
            newlist.append(tmp[1])
    return newlist

def reorder(list1):
    # reorder the folder with numberic names
    newlist=['']*len(list1)
    tmplist=[0]*len(list1)
    index=0
    for ll in list1:
        tmp=ll.split("-")
        tmplist[index]=int(tmp[0])
        index=index+1
    #index of list1
    ind1=[tmplist.index(x) for x in tmplist]
    #index of sorted list1
    ind2=[tmplist.index(x) for x in sorted(tmplist)]
    #reorder subfolder list
    for ii in ind1:
        newlist[ii]=list1[ind2[ii]]

    return(newlist)


def runpapy_par(argv1):
    # create a separate running folder
    if not os.path.exists(argv1):
        os.makedirs(argv1)
    os.chdir(argv1)
    shutil.copy2("../pa.py", "./pa.py")

    os.system("python ../pa.py ../TutorialData%s.csv %s 10:50:500 0.05:0.05:0.8 10 2"%(argv1,argv1))

if __name__ == "__main__":
    #get input parameters
    args = sys.argv

    # runpapy_par(args[1]) #run pa.py in separate folders

    subdirs=SubDirPath(".")
    print(type(subdirs))
    newlist=patternFilter(subdirs)
    print(newlist)

    newlist1=reorder(newlist)
    print(newlist1)

    output_dir="results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pa_methods=['diffgroups-','linearregression-']
    pa_opts = ['fpn', 'fpb', 'fpbh', 'fpby','tnn', 'tnb', 'tnbh', 'tnby', \
                             'fdn', 'fdb', 'fdbh', 'fdby', 'fnn', 'fnb', 'fnbh', 'fnby', \
                             'sfdn', 'sfdb', 'sfdbh', 'sfdby', 'stpn', 'stpb', 'stpbh', 'stpby', \
                             'stnn', 'stnb', 'stnbh', 'stnby', 'sfnn', 'sfnb', 'sfnbh', 'sfnby', \
                             'sfpn', 'sfpb', 'sfpbh', 'sfpby', 'tpn', 'tpb', 'tpbh', 'tpby']

    for mm in pa_methods:
        for op in pa_opts:
            savefilename = "%s/%s%s.csv" % (output_dir, mm, op)
            for ll in newlist1:
                srcfilename="%s/papy_output/%s%s.csv"%(ll,mm,op)
                if not os.path.exists(savefilename):
                    shutil.copy2(srcfilename, savefilename)
                else:
                    tmpData=pd.read_csv(srcfilename)
                    file_handle = file(savefilename,'a')
                    np.savetxt(file_handle,np.array(tmpData), delimiter=",", fmt='%.5f')
                    file_handle.close()
            tmpData1=pd.read_csv(savefilename)
            cols=tmpData1.shape[1]
            col_names=tmpData1.columns[range(1,cols)]
            tmpData1=tmpData1[col_names]
            tmpData2=tmpData1.groupby('Effect Sizes (Sample Sizes in Columns)', as_index=False).mean()
            savefilename1="%s/mean-%s%s.csv" % (output_dir, mm, op)
            tmpData2.to_csv(savefilename1, index=False)

    diff_median_files = ['output_uncTP_ratio_median', 'output_bonfTP_ratio_median', 'output_bhTP_ratio_median',
                           'output_byTP_ratio_median']
    ln_median_files = ['output_uncTP_ratio_median_ln', 'output_bonfTP_ratio_median_ln',
                           'output_bhTP_ratio_median_ln', 'output_byTP_ratio_median_ln']

    for ff in diff_median_files:
        diff_median_tmp = pd.DataFrame()
        for ll in newlist1:
            srcfilename="%s/papy_output/%s.csv"%(ll,ff)
            if os.path.exists(srcfilename):
                tmp=pd.read_csv(srcfilename)
                diff_median_tmp=diff_median_tmp.append(tmp)
                #print(diff_median_tmp.size)
        if not diff_median_tmp.size==0:
            diff_median_tmp = diff_median_tmp.groupby('Effect Sizes (Sample Sizes in Columns)', as_index=False).median()
            savefilename1="%s/%s.csv"%(output_dir, ff)
            diff_median_tmp.to_csv(savefilename1, index=False)
    for ff in ln_median_files:
        ln_median_tmp = pd.DataFrame()
        for ll in newlist1:
            srcfilename="%s/papy_output/%s.csv"%(ll,ff)
            if os.path.exists(srcfilename):
                tmp = pd.read_csv(srcfilename)
                ln_median_tmp=ln_median_tmp.append(tmp)
        if not ln_median_tmp.size==0:
            ln_median_tmp = ln_median_tmp.groupby('Effect Sizes (Sample Sizes in Columns)', as_index=False).median()
            savefilename1="%s/%s.csv"%(output_dir, ff)
            ln_median_tmp.to_csv(savefilename1, index=False)
    #plot surface plot for mean of power calculation
    for mm in pa_methods:
        for op in pa_opts:
            file_to_plot="%s/mean-%s%s.csv" % (output_dir, mm, op)
            if os.path.exists(file_to_plot):
                tmpData=pd.read_csv(file_to_plot)
                data_to_plot=np.array(tmpData[tmpData.columns[1:]])
                data_to_plot = np.expand_dims(data_to_plot,axis=0)
                data_to_plot = np.expand_dims(data_to_plot, axis=0)
                data_to_plot = np.expand_dims(data_to_plot, axis=0)
                titles=tmpData.columns.values
                sampleSizes = titles[1:].astype('int')
                sampleSizes = np.expand_dims(sampleSizes, axis=0)
                effectSizes = np.array(tmpData[tmpData.columns[0]])
                effectSizes = np.expand_dims(effectSizes, axis=0)
                iSurfacePlot(data_to_plot,"%s/plot-mean-%s%s.html" % (output_dir, mm, op), 1, 1, 1, sampleSizes, effectSizes)
    #plot proportion of power > 0.8
    output_plot=['power-rate-noCorrection-diffgroups','power-rate-bonfCorrection-diffgroups', \
                 'power-rate-bhCorrection-diffgroups','power-rate-byCorrection-diffgroups']
    corr_type=['no correction','Bonferroni correction','Benjamini-Hochberg correction','Benjamini-Yekutieli correction']
    index=0
    for ff in diff_median_files:
        file_to_plot="%s/%s.csv"%(output_dir, ff)
        if os.path.exists(file_to_plot):
            tmpData = pd.read_csv(file_to_plot)
            data_to_plot = np.array(tmpData[tmpData.columns[1:]])
            titles = tmpData.columns.values
            sampleSizes = titles[1:].astype('int')
            sampleSizes = np.expand_dims(sampleSizes, axis=0)
            effectSizes = np.array(tmpData[tmpData.columns[0]])
            effectSizes = np.expand_dims(effectSizes, axis=0)
            iSurfacePlotTPR(data_to_plot, '%s/plot-%s.html'%(output_dir,output_plot[index]), corr_type[index], sampleSizes, effectSizes)
            slice_rows = np.array([1, int(floor(effectSizes.shape[1] / 2)), effectSizes.shape[1] - 2])
            Y_temp = []
            for sl in slice_rows:
                Y_temp.append(data_to_plot[sl, :])
            iSlicesPlot(sampleSizes[0], Y_temp, \
                        '%s/plot-slice-%s.html'%(output_dir,output_plot[index]), \
                        'Proportion of Variables with Power (True Positive)> 0.8 -(%s)'%(corr_type[index]), \
                        'Sample Size', 'Proportion', 'Effect Size=', effectSizes[:, slice_rows])
            slice_cols = np.array([1, int(floor(sampleSizes.shape[1] / 2)), sampleSizes.shape[1] - 2])
            Y_temp = []
            for sl in slice_cols:
                Y_temp.append(data_to_plot[:,sl])
            iSlicesPlot(effectSizes[0], Y_temp, \
                        '%s/plot-slice-%s-eff.html'%(output_dir,output_plot[index]), \
                        'Proportion of Variables with Power (True Positive)> 0.8 -(%s)'%(corr_type[index]), \
                        'Effect Size', 'Proportion', 'Sample Size=', sampleSizes[:, slice_cols])
        index=index+1

    index = 0
    for ff in ln_median_files:
        file_to_plot = "%s/%s.csv" % (output_dir, ff)
        if os.path.exists(file_to_plot):
            tmpData = pd.read_csv(file_to_plot)
            data_to_plot = np.array(tmpData[tmpData.columns[1:]])
            titles = tmpData.columns.values
            sampleSizes = titles[1:].astype('int')
            sampleSizes = np.expand_dims(sampleSizes, axis=0)
            effectSizes = np.array(tmpData[tmpData.columns[0]])
            effectSizes = np.expand_dims(effectSizes, axis=0)
            iSurfacePlotTPR(data_to_plot, '%s/plot-%s-linearregression.html' % (output_dir, output_plot[index]), corr_type[index],
                            sampleSizes, effectSizes)
            slice_rows = np.array([1, int(floor(effectSizes.shape[1] / 2)), effectSizes.shape[1] - 2])
            Y_temp = []
            for sl in slice_rows:
                Y_temp.append(data_to_plot[sl, :])
            iSlicesPlot(sampleSizes[0], Y_temp, \
                        '%s/plot-slice-%s-linearregression.html' % (output_dir, output_plot[index]), \
                        'Proportion of Variables with Power (True Positive)> 0.8 -(%s)' % (corr_type[index]), \
                        'Sample Size', 'Proportion', 'Effect Size=', effectSizes[:, slice_rows])
            slice_cols = np.array([1, int(floor(sampleSizes.shape[1] / 2)), sampleSizes.shape[1] - 2])
            Y_temp = []
            for sl in slice_cols:
                Y_temp.append(data_to_plot[:, sl])
            iSlicesPlot(effectSizes[0], Y_temp, \
                        '%s/plot-slice-%s-linearregression-eff.html' % (output_dir, output_plot[index]), \
                        'Proportion of Variables with Power (True Positive)> 0.8 -(%s)' % (corr_type[index]), \
                        'Effect Size', 'Proportion', 'Sample Size=', sampleSizes[:, slice_cols])
        index = index + 1