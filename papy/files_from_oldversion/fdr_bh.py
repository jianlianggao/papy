## function of false discovery rate Benjamini & Hochberg FDR_BH
import sys,inspect,dis
import numpy as np

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

if __name__=="__main__":    
    fdr_bh(pvals, q=0.05, method='pdep', report='no')