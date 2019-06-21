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
    Pa, pix = Pareto(X)
    ide = ideal(Pa)
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
    distance = dist(M,pref) 
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
    pref = np.ones( shape=(m,) )/m
    return X, pref

def norm(X):
    mn = np.min( X, axis=0 )
    mx = np.max( X, axis=0 )
    return (X-mn)/(mx-mn)

# In[]:
    
X, pref = init( 150,6 )

Xn = regularization(X)
Pa = Pareto(Xn)
Fr,fix = fronts(Xn)
ranking = rank(Xn)

# In[]:
""" Example visualization for a 2-D case """

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

X, pref = init( 20,2 )
pref = np.random.random(len(pref))
pref = pref/sum(pref)

# In[]:


Xn = regularization(X)
Pa = Pareto(Xn)
Fr,fix = fronts(Xn)
ranking = rank(Xn,pref)

def plotRanking(X, fix, ranking):
    """ Plots the Pareto fronts and the ranking.
    PCA dimensionality reduction is used if d > 2 """
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
    plt.xlabel('Criteria 1')
    plt.ylabel('Criteria 2')
    plt.scatter(0,0,c='red',marker='*')
    plt.text(0,0,'ideal',color='red')
    plt.scatter(1,1,c='blue',marker='*')
    plt.text(1,1,'nadir',color='blue')
    plt.plot()
    
plotRanking(Xn, fix, ranking)
