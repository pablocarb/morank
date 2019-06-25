#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MORank (c) University of Manchester 2019

https://github.com/pablocarb/morank.git is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

Created on Thu Jun 20 16:28:07 2019

@author:  Pablo Carbonell, SYNBIOCHEM
@description: Multiobjective ranking 
"""

import numpy as np

def dominate( A,B,obj='min'):
    """ Check if A dominates B """
    if obj == 'max':
        return np.all( np.greater_equal(A,B) )  & np.any( np.greater(A,B) )
    else:
        return np.all( np.less_equal(A,B) )  & np.any( np.less(A,B) )
    
def ideal( par,obj='min' ):
    """ Get ideal objective vector in the Pareto front """
    if obj == 'max':
        return np.max( par, axis=0 )
    else:
        return np.min( par, axis=0 )
    
def nadir( X,obj='min' ):
    if obj == 'max':
        return np.min( X,axis=0 )
    else:
        return np.max( X,axis=0 )
    
def normalize( X, ide, nad ):
    return (X-ide)/(nad-ide)


def regularization(X):
    Pa = Pareto(X)
    ide = ideal(Pa[0])
    nad = nadir(X)
    Xn = normalize( X,ide,nad )
    return Xn  

def Pareto(M):
    """ Calculate the set of non-dominated solution """
    # Kept first solution
    Pp = set( [0] )
    # For each other solution 
    for i in np.arange(1,M.shape[0]):
        # Compare with all members of Pp
        nondom = True
        for j in list(Pp):
            dom = dominate( M[i,], M[j,] )
            # If i dominates j, remove j
            if dom:
                Pp.remove(j)
            # Check if j dominates i
            dom = dominate( M[j,], M[i,] )
            if dom:
                nondom = False
                break
            # If no j dominates i, add i
        if nondom:
            Pp.add(i)
    ix = sorted(list(Pp))
    return M[ix,:], ix

def fronts(M):
    """ Compute the Pareto fronts """
    fr = []
    ixs = []
    fix = np.arange(0,M.shape[0])
    ix = np.arange(0,M.shape[0])
    X = M
    while len(ix) > 0:
        PA, pix = Pareto(X)
        fr.append( PA )
        ixs.append( fix[pix] )
        ix = list( set(np.arange(0,X.shape[0])) - set(pix) )
        fix = fix[ix]
        X = X[ix,:]
    return fr, ixs

def dist( M,x=0,p=2,pref=None ):
    if pref is not None:
        return np.power( np.sum( ((M-x)**p)*pref, axis=1), 1/p ) 
    else:
        return np.power( np.sum( (M-x)**p, axis=1), 1/p ) 
        
    
def rank(M,pref=None):
    """ Rank solutions:
        - By Pareto front subpopulation
        - Distance to ideal vector
    """
    distance = dist(M,pref=np.abs(pref)) 
    fr, fix = fronts(M)
    strata = np.zeros(shape=len(distance),dtype=int)
    for i in np.arange(0,len(fix)):
        for j in fix[i]:
            strata[j] = i
    cols = np.array(list(zip(strata,distance)),dtype=[('x',float),('y',float)])
    ranking = np.argsort( cols,axis=0 )
    return ranking
                
    
# In[ ]:

def init(n=50,m=8):
    """ Random init of criteria matrix:
        n = number of solutions (pathways);
        m = number of criteria """
        
    X = np.random.normal( size=(n,m) )
    return X

def norm(X):
    mn = np.min( X, axis=0 )
    mx = np.max( X, axis=0 )
    return (X-mn)/(mx-mn)

def sanitize(X,pref):
    """ Ignore constant and zero-pref columns
    and convert all objectives to minimization"""
    mask = np.sign( np.negative( pref ) )
    M = np.array(X)*mask
    ix = np.where( np.max(M,axis=0) - np.min(M,axis=0) > 0)[0]
    M = M[:,ix]
    pref = np.array(pref)[ix]
    return M, pref
     
def rankPaths(X,pref):
    """ Rank pathways
        - X : decision matrix;
        - pref : preference weights (sign>1 for maximize, sign<1 for minimize)
    """
    # Set minimization for all objectives
    X, pref = sanitize(X,pref)
    # Regularize in [0,1]
    Xn = regularization(X)
    # Rank pathways
    ranking = rank(Xn,pref)
    return ranking
    

# In[]:

def example(): 
    n = 150
    m = 8
    pref = np.random.uniform(low=-1,high=1,size=m)
    pref[0] = 0
    pref = pref/sum(pref)
    X = init( n,m )
    ranking = rankPaths(X,pref)
    return(ranking)

# In[]:
""" Example visualization for a 2-D case """

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plotRanking(Xn, pref, ranking):
    """ Plots the Pareto fronts and the ranking.
    PCA dimensionality reduction is used if d > 2 """
    X = np.array(Xn)
    minpref = np.negative( pref )
    mask = np.sign( minpref )
    Fr,fix = fronts(X*mask)
    if X.shape[1] > 2:       
        pca = PCA(n_components=2)
        pca.fit(X)
        Xp = pca.transform(X)
    else:
        Xp = X
    plt.clf()
    plt.scatter(Xp[:,0],Xp[:,1])
    for i in np.arange(0,len(fix)):
        ix1 = np.argsort( Xp[fix[i],0] )
        plt.plot(Xp[fix[i][ix1],0],Xp[fix[i][ix1],1])
    for i in np.arange(0,len(ranking)):
        j = ranking[i]
        plt.text(Xp[j,0],Xp[j,1],str(i))
    try:
        plt.xlabel(Xn.columns[0])
        plt.ylabel(Xn.columns[1])
    except:
        plt.xlabel('Criterion 1')
        plt.ylabel('Criterion 2')
    mask = np.sign(np.negative(pref))
    ide = ideal(X*mask)*mask
    nad = nadir(X*mask)*mask
    plt.scatter(ide[0],ide[1],c='red',marker='*')
    plt.text(ide[0],ide[1],'ideal',color='red')
    plt.scatter(nad[0],nad[1],c='blue',marker='*')
    plt.text(nad[0],nad[1],'nadir',color='blue')
    plt.plot()
    

#def example2d():
if __name__ == '__main__':
    n = 50
    titer = np.random.uniform(size=50)
    thermo = np.random.normal(-5,4,size=50)
    X = pd.DataFrame({'Thermo':thermo,'Titer':titer})
    pref = [-0.2,0.5]
    ranking = rankPaths(X,pref)
    plotRanking(X, pref, ranking)
    
    # In[]:
    
    

