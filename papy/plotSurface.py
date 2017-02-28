#!/usr/bin/env python
"""
Plotting Power Proportion and False Negative Proportion 
for Power Analysis (calculation) tool
Developed by Dr Jianliang Gao
Imperial College London
2016
"""
import sys
import pandas as pd             ##for reading data from csv files
import numpy as np

def iSurfacePlot(results, svfilename, variable, sizeeff,samplsizes):
    import plotly as py
    import plotly.graph_objs as go

    MUtot = np.array(results)
    NS, NSE = MUtot.shape
    
    ##plot
    ##generate a 2D grid
    X, Y = np.meshgrid(sizeeff, samplsizes)
    
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
    py.offline.plot(fig, filename="z_plots_%s_variable_%i.html"%(svfilename,variable), auto_open=False)
    print("Your plot is saved as z_plots_%s_variable_%i.html"%(svfilename,variable))

def main(argv1, argv2, argv3):
    print(argv1)
    print(argv2)
    print(argv3)
    data = pd.read_csv("%s-%s.csv"%(argv1, argv2))
    titles=list(data.columns.values)
    output=data.loc[data['Variables']==int(argv3)]
    results=output.loc[:,titles[2]:]
    variable1=data.loc[data['Variables']==1]
    effect_size=np.array(variable1.loc[:, titles[1]])
    sample_size=titles[2:]
    sample_size=np.array(list(map(int, sample_size)))
    iSurfacePlot(results, argv2, int(argv3), sample_size,effect_size)

if __name__=="__main__":
    try:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    except:
        print('usage: python plotSurface.py arg1 arg2 arg3')
        print('for example: python plotSurface.py diffgroups fnb 8 ')
        print('--arg1 can be \'linearregression\' or \'diffgroups\' ')
        print('--arg2 can be one of the following combination from:')
        print('[tp,  fn, tn, fp]  and [ n, b, bh, by]')
        print('for example tpn, means True Positive (no correction)')
        print('--arg3 is an integer as number of variables')