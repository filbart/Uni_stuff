#!/usr/bin/env python
# -*- coding: utf-8 -*-
#one of the COCO code files, optimizer functions added by FWB
#files for the biologically inspired computation
#FWBARTOSZEWSKI & ANGUS HAMILTON
"""Runs an entire experiment for benchmarking PURE_RANDOM_SEARCH on a testbed.

ADDED OPTIMIZER FUNCTIONS BY FWBARTOSZEWSKI

CAPITALIZATION indicates code adaptations to be made.
This script as well as files bbobbenchmarks.py and fgeneric.py need to be
in the current working directory.

Under unix-like systems: 
    nohup nice python exampleexperiment.py [data_path [dimensions [functions [instances]]]] > output.txt &

"""
import sys # in case we want to control what to run via command line args
import time
import numpy as np
import fgeneric
import bbobbenchmarks

argv = sys.argv[1:] # shortcut for input arguments

datapath = 'PUT_MY_BBOB_DATA_PATH' if len(argv) < 1 else argv[0]

dimensions = (2, 3, 5, 10, 20, 40) if len(argv) < 2 else eval(argv[1])
function_ids = bbobbenchmarks.nfreeIDs if len(argv) < 3 else eval(argv[2])  
# function_ids = bbobbenchmarks.noisyIDs if len(argv) < 3 else eval(argv[2])
instances = range(1, 6) + range(41, 51) if len(argv) < 4 else eval(argv[3])

opts = dict(algid='PUT ALGORITHM NAME',
            comments='PUT MORE DETAILED INFORMATION, PARAMETER SETTINGS ETC')
maxfunevals = '10 * dim' # 10*dim is a short test-experiment taking a few minutes 
# INCREMENT maxfunevals SUCCESSIVELY to larger value(s)
minfunevals = 'dim + 2'  # PUT MINIMAL sensible number of EVALUATIONS before to restart
maxrestarts = 10000      # SET to zero if algorithm is entirely deterministic 


def run_optimizer(fun, dim, maxfunevals, ftarget=-np.Inf):
    """start the optimizer, allowing for some preparation. 
    This implementation is an empty template to be filled 
    
    """
    # prepare
    x_start = 8. * np.random.rand(dim) - 4
    
    # call, REPLACE with optimizer to be tested
    PURE_SMART_TOURNAMENT(fun, x_start, maxfunevals, ftarget)

def PURE_RANDOM_SEARCH(fun, x, maxfunevals, ftarget):
    """samples new points uniformly randomly in [-5,5]^dim and evaluates
    them on fun until maxfunevals or ftarget is reached, or until
    1e8 * dim function evaluations are conducted.
    original COCO optimizer

    """
    dim = len(x)
    maxfunevals = min(1e8 * dim, maxfunevals)
    popsize = min(maxfunevals, 200)
    fbest = np.inf
    
    for _ in range(0, int(np.ceil(maxfunevals / popsize))):
        xpop = 10. * np.random.rand(popsize, dim) - 5.
        fvalues = fun(xpop)
        idx = np.argsort(fvalues)
        if fbest > fvalues[idx[0]]:
            fbest = fvalues[idx[0]]
            xbest = xpop[idx[0]]
        if fbest < ftarget:  # task achieved 
            break

    return xbest
	
    
def GENERATIONAL_FANCY(fun, x, maxfunevals, ftarget):
    """creates random pairs from the whole population and then their children
    created using k-point crossover take place of almost the whole population
    only the best chromosome of the previous generation survives
    """
    dim = len(x)
    maxfunevals = min(1e8 * dim, maxfunevals)
    popsize = min(maxfunevals, 200)
    fbest = np.inf
	
    xpop = 10. * np.random.rand(popsize, dim) - 5.
    fvalues = fun(xpop)
    idx = np.argsort(fvalues)

    for iter in range(0, int(np.ceil(maxfunevals / popsize))):
        for j in range(0,int(2*(popsize-1))):
            xpairs=[]
#creating pairs of parents
            (h1,h2)=np.random.choice(len(xpop),2)
            h1=xpop[h1]
            h2=xpop[h2]
#finding the worst chromosome
            old=min(fvalues)
            for i in xpop:
                if fun(i)==old:
                    old=i
                    break
            xpairs.append((h1,h2))
#crossover happening
            k=np.random.choice(range(dim))
            count=0
            for pair in xpairs:
                for i in range(0,dim):
                    helper=[]
                    if i<k:
					helper.append(pair[0][i])
                    else:
					helper.append(pair[1][i])
#updating the population
                xpop[count]=helper
                count+=1
            xpop[-1]=old
            fvalues = fun(xpop)
            idx = np.argsort(fvalues)
            if fbest > fvalues[idx[0]]:
                fbest = fvalues[idx[0]]
                xbest = xpop[idx[0]]
            if fbest < ftarget:  # task achieved 
                break

    return xbest
    
def MUTANT_GENERATION(fun, x, maxfunevals, ftarget):
    """similiar to FANCY algorithm, but we use uniform crossover for the children
    and then we mutate them, so we change our genepool slightly with each generation    
    """
    dim = len(x)
    maxfunevals = min(1e8 * dim, maxfunevals)
    popsize = min(maxfunevals, 200)/2
    fbest = np.inf
	
    xpop = 10. * np.random.rand(popsize, dim) - 5.
    fvalues = fun(xpop)
    idx = np.argsort(fvalues)

    for iter in range(0, int(np.ceil(maxfunevals / popsize))):
        for j in range(0,int(2*(popsize-1))):
#grabbin pairs of parents
            xpairs=[]
            (h1,h2)=np.random.choice(len(xpop),2)
            h1=xpop[h1]
            h2=xpop[h2]
            h0=h1
#finding the worst individual
            old=min(fvalues)
            for i in xpop:
                if fun(i)==old:
                    old=i
                    break
            xpairs.append((h1,h2))
            count=0
#crossover and mutation happening
            for pair in xpairs:
                for i in range(dim):
                    if (np.random.choice((1,2)))==2:
                        h0[i]=h2[i]
                helper1=np.random.normal(h0,0.3)
                while np.all(helper1.all()<-5 or helper1.all()>5):
                    helper1=np.random.normal(h0,0.3)
#updating the population
                xpop[count]=helper1
                count+=1
            xpop[-1]=old
            fvalues = fun(xpop)
            idx = np.argsort(fvalues)
            if fbest > fvalues[idx[0]]:
                fbest = fvalues[idx[0]]
                xbest = xpop[idx[0]]
            if fbest < ftarget:  # task achieved 
                break

    return xbest
    
def PURE_HILLCLIMB(fun, x, maxfunevals, ftarget):
    """ we've got a single point that tries to climb it's way towards minimum of the function
	it mutates using gaussian distribution
    """
    dim = len(x)
    maxfunevals = min(1e8 * dim, maxfunevals)
    popsize = 1
    fbest = np.inf
    xpop = 10. * np.random.rand(popsize, dim) - 5.
    for iter in range(0, int(np.ceil(maxfunevals / popsize))):
#mutating the single point, checkinh if it's still in the area
        xpopnew=np.random.normal(xpop,0.3)
        while np.all(xpopnew.all()<-5 or xpopnew.all()>5):
            xpopnew=np.random.normal(xpop,0.3)
#checking if new fitness is better
        if fun(xpop)<fun(xpopnew):
		xpop=xpopnew
        fvalues=fun(xpop)
        if fbest > fvalues:
            fbest = fvalues
            xbest = xpop
        if fbest < ftarget:  # task achieved 
            break
    return xbest
	
	
def PURE_SMART_TOURNAMENT(fun, x, maxfunevals, ftarget):
    """random population, we have a tournament between two random points,
	the looser is replaced with a child of the winner
     uses normal distribution for the child, checks fitness for each new gene
    """
    dim = len(x)
    maxfunevals = min(1e8 * dim, maxfunevals)+700
    popsize = min(maxfunevals, 200)-20
    fbest = np.inf
    xpop = 10. * np.random.rand(popsize, dim) - 5.
    fvalues = fun(xpop)
    
    for iter in range(0, int(np.ceil(maxfunevals / popsize))):
#tournament happens
        x1,x2=np.random.choice(len(xpop),2)
        x1=xpop[x1]
        x2=xpop[x2]
#winner becomes parent, looser will be replaced by the child
        if fun(x1)<fun(x2):
            par=x1
            old=fun(x2)
        else:
            par=x2
            old=fun(x1)
        counter=0
        for i in xpop:
            if fun(i)==old:
                old=counter
                break
            counter+=1
        helper1=par
        helper2=par
        counter=0
#mutation happens, each gene is separately checked if mutating it would be useful
        for i in par:
            helper1[counter]=np.random.normal(i,0.2)
            if fun(helper1)<fun(par):
                helper2=helper1
            counter+=1
        while helper2.any()<-5. or helper2.any()>5.:
            helper1=par
            helper2=par
            counter=0
            for i in par:
                helper1[counter]=np.random.normal(i,0.2)
                if fun(helper1)<fun(par):
                    helper2=helper1
                counter+=1
#if the child has better fitness it replaces the looser
        if fun(helper2)<fun(xpop[old]):
            xpop[old]=helper2
        fvalues = fun(xpop)
        idx = np.argsort(fvalues)
        if fbest > fvalues[idx[0]]:
            fbest = fvalues[idx[0]]
            xbest = xpop[idx[0]]
        if fbest < ftarget:  # task achieved 
            break

    return xbest
	
def PURE_UNIFORM_CROSSOVER(fun, x, maxfunevals, ftarget):
    """population of random points, we grab two random from it,
    the better one becomes a parent. the child
    takes place of the worst specimen in the population, 
    child is determined using uniform crossover
    """
    dim = len(x)
    maxfunevals = min(1e8 * dim, maxfunevals)
    popsize = min(maxfunevals, 200)
    fbest = np.inf
	
    xpop = 10. * np.random.rand(popsize, dim) - 5.

    for iter in range(0, int(np.ceil(maxfunevals / popsize))):
        fvalues = fun(xpop)
#we grab a pair of parents
        h1,h2=np.random.choice(len(xpop),2)
        x1=xpop[h1]
        x2=xpop[h2]
        helper=x1
        if fun(x1)>fun(x2):
            old=h1
        else:
            old=h2
#crossover happens
        for i in range(dim):
            h1=(np.random.choice((1,2)))
            if h1==2:
                helper[i]=x2[i]
        xpop[old]=helper
        fvalues = fun(xpop)
        idx = np.argsort(fvalues)
        if fbest > fvalues[idx[0]]:
            fbest = fvalues[idx[0]]
            xbest = xpop[idx[0]]
        if fbest < ftarget:  # task achieved 
            break

    return xbest
    
def MUTATING_UNIFORM_CROSSOVER(fun, x, maxfunevals, ftarget):
    """population of random points, we grab two random from it,
    the better one becomes a parent. the child
    takes place of the worst specimen in the population, 
    child is determined using uniform crossover
    """
    dim = len(x)
    maxfunevals = min(1e8 * dim, maxfunevals)
    popsize = min(maxfunevals, 200)
    fbest = np.inf
	
    xpop = 10. * np.random.rand(popsize, dim) - 5.

    for iter in range(0, int(np.ceil(maxfunevals / popsize))):
        fvalues = fun(xpop)
#we grab a pair of parents
        h1,h2=np.random.choice(len(xpop),2)
        x1=xpop[h1]
        x2=xpop[h2]
        helper=x1
        if fun(x1)>fun(x2):
            old=h1
        else:
            old=h2
#crossover and mutation happens
        for i in range(dim):
            h1=(np.random.choice((1,2)))
            if h1==2:
                helper[i]=x2[i]
        helper1=np.random.normal(helper,0.3)
        while np.all(helper1.all()<-5 or helper1.all()>5):
            helper1=np.random.normal(helper,0.3)
        xpop[old]=helper1
        fvalues = fun(xpop)
        idx = np.argsort(fvalues)
        if fbest > fvalues[idx[0]]:
            fbest = fvalues[idx[0]]
            xbest = xpop[idx[0]]
        if fbest < ftarget:  # task achieved 
            break

    return xbest
	
def PURE_ONE_POINT_CROSSOVER(fun, x, maxfunevals, ftarget):
    """population of random points, we grab two random from it,
    the better one becomes a parent. the child
    takes place of the worst specimen in the population, 
    child is determined using one point crossover
    """
    dim = len(x)
    maxfunevals = min(1e8 * dim, maxfunevals)
    popsize = min(maxfunevals, 200)
    fbest = np.inf
	
    xpop = 10. * np.random.rand(popsize, dim) - 5.
    

    for iter in range(0, int(np.ceil(maxfunevals / popsize))):
        fvalues = fun(xpop)
#grabbing parents
        h1,h2=np.random.choice(len(xpop),2)
        x1=xpop[h1]
        x2=xpop[h2]
        k=np.random.choice(range(dim))
        helper=x1
        if fun(x1)>fun(x2):
            old=h1
        else:
            old=h2
#crossover happens
        for i in range(dim):
            if i>k:
                helper[i]=x2[i]
        xpop[old]=helper
        fvalues = fun(xpop)
        idx = np.argsort(fvalues)
        if fbest > fvalues[idx[0]]:
            fbest = fvalues[idx[0]]
            xbest = xpop[idx[0]]
        if fbest < ftarget:  # task achieved 
            break

    return xbest
    
    
def PURE_FILIP_SEARCH(fun, x, maxfunevals, ftarget):
    """we create a random population, choose two random chromosomes,
    the better one becomes a parent, child replaces the worst chromosome
    in the whole pool, mutation uses normal distribution
    """
    dim = len(x)
    maxfunevals = min(1e8 * dim, maxfunevals)
    popsize = min(maxfunevals, 200)
    fbest = np.inf
	
    xpop = 10. * np.random.rand(popsize, dim) - 5.
    fvalues = fun(xpop)

    for iter in range(0, int(np.ceil(maxfunevals / popsize))):
#we grab two random chromosomes, a better one becomes a parent
        x1=np.random.choice(xpop)
        x2=np.random.choice(xpop)
        if fun(x1)<fun(x2):
		par=x1
        else:
            par=x2
        old=max(fvalues)
#mutation, worst individual becomes the child
        xpop[old]=np.random.normal(par,1)
        while xpop[old]>-5 and xpop<5:
            xpop[old]=np.random.normal(par,1)
        fvalues = fun(xpop)
        idx = np.argsort(fvalues)
        if fbest > fvalues[idx[0]]:
            fbest = fvalues[idx[0]]
            xbest = xpop[idx[0]]
        if fbest < ftarget:  # task achieved 
            break

    return xbest
#actual program code
t0 = time.time()
np.random.seed(int(t0))

f = fgeneric.LoggingFunction(datapath, **opts)
for dim in dimensions:  # small dimensions first, for CPU reasons
    for fun_id in function_ids:
        for iinstance in instances:
            f.setfun(*bbobbenchmarks.instantiate(fun_id, iinstance=iinstance))

            # independent restarts until maxfunevals or ftarget is reached
            for restarts in xrange(maxrestarts + 1):
                if restarts > 0:
                    f.restart('independent restart')  # additional info
                run_optimizer(f.evalfun, dim,  eval(maxfunevals) - f.evaluations,
                              f.ftarget)
                if (f.fbest < f.ftarget
                    or f.evaluations + eval(minfunevals) > eval(maxfunevals)):
                    break

            f.finalizerun()

            print('  f%d in %d-D, instance %d: FEs=%d with %d restarts, '
                  'fbest-ftarget=%.4e, elapsed time [h]: %.2f'
                  % (fun_id, dim, iinstance, f.evaluations, restarts,
                     f.fbest - f.ftarget, (time.time()-t0)/60./60.))

        print '      date and time: %s' % (time.asctime())
    print '---- dimension %d-D done ----' % dim
