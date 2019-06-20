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

def comprorank(M, p=2):
    """ Global criterion or compromise programming ranking.
        p: norm
    """
    X = norm( M )
    if p == 0:
        W = np.max( (1-X), axis=1 )
    else:
        W = np.power( np.sum( (1-X)**p, axis=1), 1/p ) 
    rank = np.argsort( W )
    return rank
    
# In[ ]:

def init(n=50,m=8):
    """ Random init of criteria matrix:
        n = number of solutions (pathways);
        m = number of criteria """
        
    X = np.random.random(size=(n,m))
    return X

def norm(X):
    mn = np.min( X, axis=0 )
    mx = np.max( X, axis=0 )
    return (X-mn)/(mx-mn)