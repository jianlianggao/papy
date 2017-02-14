## function - SurfacePlot

import numpy as np
import matplotlib.pyplot as plt

# For 3d plots. This import is necessary to have 3D plotting below
from mpl_toolkits.mplot3d import Axes3D

# for saving the plot to pdf file 
from matplotlib.backends.backend_pdf import PdfPages

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
    
if __name__=="__main__":
    SurfacePlot(output, variable,metric,correction, sizeeff,samplsizes,nreps)