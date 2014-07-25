import logging, time, argparse
from numpy import dot, zeros, kron, array, eye, ones, savetxt, loadtxt
from numpy.linalg import qr, pinv, norm, inv 
from numpy.random import rand
from scipy import sparse
from scipy.sparse import coo_matrix, lil_matrix
from scipy.sparse.linalg import eigsh
import numpy as np
import os
import fnmatch
import sys
from commonFunctions import squareFrobeniusNormOfSparse, squareFrobeniusNormOfSparse2, fitNormWithoutNormX

import time
import threading


__DEF_MAXITER = 50
__DEF_PREHEATNUM = 1
__DEF_INIT = 'nvecs'
__DEF_PROJ = True
__DEF_CONV = 1e-5
__DEF_LMBDA = 0


def calc_norm(m, i, norms):

    print 'Starting thread..', i

    norm = squareFrobeniusNormOfSparse2(m)
    norms[i] = norm

def precompute_norms(X):
    print >> sys.stderr, "Precomputing Norms ..."
    norms = [0] * len(X)

    threads = []
    for i, m in enumerate(X):

        t = threading.Thread(target=calc_norm, args=(m, i, norms))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print >> sys.stderr, "Precomputed Norms."
    
    #print norms
    
    #exit (0)
    
    
    
    return norms

def rescal(X, rank, **kwargs):
    """
    RESCAL 

    Factors a three-way tensor X such that each frontal slice 
    X_k = A * R_k * A.T. The frontal slices of a tensor are 
    N x N matrices that correspond to the adjacency matrices 
    of the relational graph for a particular relation.

    For a full description of the algorithm see: 
      Maximilian Nickel, Volker Tresp, Hans-Peter-Kriegel, 
      "A Three-Way Model for Collective Learning on Multi-Relational Data",
      ICML 2011, Bellevue, WA, USA

    Parameters
    ----------
    X : list
        List of frontal slices X_k of the tensor X. The shape of each X_k is ('N', 'N')
    rank : int 
        Rank of the factorization
    lmbda : float, optional 
        Regularization parameter for A and R_k factor matrices. 0 by default 
    init : string, optional
        Initialization method of the factor matrices. 'nvecs' (default) 
        initializes A based on the eigenvectors of X. 'random' initializes 
        the factor matrices randomly.
    proj : boolean, optional 
        Whether or not to use the QR decomposition when computing R_k.
        True by default 
    maxIter : int, optional 
        Maximium number of iterations of the ALS algorithm. 50 by default. 
    conv : float, optional 
        Stop when residual of factorization is less than conv. 1e-5 by default        

    Returns 
    -------
    A : ndarray 
        matrix of latent embeddings A
    R : list
        list of 'M' arrays of shape ('rank', 'rank') corresponding to the factor matrices R_k 
    f : float 
        function value of the factorization 
    iter : int 
        number of iterations until convergence 
    exectimes : ndarray 
        execution times to compute the updates in each iteration
    """

    _log = logging.getLogger('RESCAL') # added by myself, as global variables
    # init options
    ainit = kwargs.pop('init', __DEF_INIT)
    proj = kwargs.pop('proj', __DEF_PROJ)
    maxIter = kwargs.pop('maxIter', __DEF_MAXITER)
    conv = kwargs.pop('conv', __DEF_CONV)
    lmbda = kwargs.pop('lmbda', __DEF_LMBDA)
    preheatnum = kwargs.pop('preheatnum', __DEF_PREHEATNUM)

    if not len(kwargs) == 0:
        raise ValueError( 'Unknown keywords (%s)' % (kwargs.keys()) )
   
    sz = X[0].shape
    dtype = X[0].dtype 
    n = sz[0]
    
    _log.debug('[Config] rank: %d | maxIter: %d | conv: %7.1e | lmbda: %7.1e' % (rank, 
        maxIter, conv, lmbda))
    
    # precompute norms of X 
    print 'precompute norms of X...'
    #normX = [squareFrobeniusNormOfSparse(M) for M in X]
    normX=precompute_norms(X)
    sumNormX = sum(normX)
    _log.debug('[Algorithm] The tensor norm: %.5f' % sumNormX)
    
    # initialize A
    print 'initialize A...'
    if ainit == 'random':
        _log.debug('[Algorithm] The random initialization will be performed.')
        A = array(rand(n, rank), dtype=np.float64)    
    elif ainit == 'nvecs':
        _log.debug('[Algorithm] The eigenvector based initialization will be performed.')
        avgX = lil_matrix((n, n))
        for i in range(len(X)):
            avgX += (X[i] + X[i].T)
        eigvals, A = eigsh(avgX, rank) 
    else :
        raise 'Unknown init option ("%s")' % ainit 

    # initialize R
    print 'initialize R...'
    if proj:
        Q, A2 = qr(A)
        X2 = __projectSlices(X, Q)
        R = __updateR(X2, A2, lmbda)
    else :
        raise 'Projection via QR decomposition is required; pass proj=true'

    _log.debug('[Algorithm] Finished initialization.')
    # compute factorization
    print 'compute factorization...'
    fit = fitchange = fitold = 0
    exectimes = []
    
    for iterNum in xrange(maxIter):
        print str(iterNum)+'th iteration...'
        tic = time.clock()
        
        A = updateA(X, A, R, lmbda)
        if proj:
            Q, A2 = qr(A)
            X2 = __projectSlices(X, Q)
            R = __updateR(X2, A2, lmbda)
        else :
            raise 'Projection via QR decomposition is required; pass proj=true'

        # compute fit values
        fit = 0
        regularizedFit = 0
        regRFit = 0 
        if iterNum >= preheatnum:
            if lmbda != 0:   
                for i in xrange(len(R)):
                    regRFit += norm(R[i])**2
                regularizedFit = lmbda*(norm(A)**2) + lmbda*regRFit
            
            for i in xrange(len(R)):
                fit += (normX[i] + fitNormWithoutNormX(X[i], A, R[i]))
            fit *= 0.5
            fit += regularizedFit
            fit /= sumNormX 
        else :
            _log.debug('[Algorithm] Preheating is going on.')        
            
        toc = time.clock()
        exectimes.append( toc - tic )
        fitchange = abs(fitold - fit)
        _log.debug('[%3d] total fit: %.10f | delta: %.10f | secs: %.5f' % (iterNum, 
        fit, fitchange, exectimes[-1]))
            
        fitold = fit
        if iterNum > preheatnum and fitchange < conv:
            break
    return A, R, fit, iterNum+1, array(exectimes)

def updateA(X, A, R, lmbda):
    n, rank = A.shape
    F = zeros((n,rank))
    E = zeros((rank, rank), dtype=np.float64)

    AtA = dot(A.T, A)
    for i in xrange(len(X)):
        ar = dot(A, R[i])
        art = dot(A, R[i].T)
        F += X[i].dot(art) + X[i].T.dot(ar)
        E += dot(R[i], dot(AtA, R[i].T)) + dot(R[i].T, dot(AtA, R[i]))
    A = dot(F, inv(lmbda * eye(rank) + E))
    return A

def __updateR(X, A, lmbda):
    r = A.shape[1]
    R = []
    At = A.T    
    if lmbda == 0:
        ainv = dot(pinv(dot(At, A)), At)
        for i in xrange(len(X)):
            R.append( dot(ainv, X[i].dot(ainv.T)) )
    else :
        AtA = dot(At, A)
        tmp = inv(kron(AtA, AtA) + lmbda * eye(r**2))
        for i in xrange(len(X)):
            AtXA = dot(At, X[i].dot(A)) 
            R.append( dot(AtXA.flatten(), tmp).reshape(r, r) )
    return R

def __projectSlices(X, Q):
    X2 = []
    for i in xrange(len(X)):
        X2.append( dot(Q.T, X[i].dot(Q)) )
    return X2

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--latent", type=int, help="number of latent components", required=True)
    parser.add_argument("--lmbda", type=float, help="regularization parameter", required=True)
    parser.add_argument("--input", type=str, help="the directory, where the input data are stored", required=True)
    parser.add_argument("--outputentities", type=str, help="the file, where the latent embedding for entities will be output", required=True)
    parser.add_argument("--outputfactors", type=str, help="the file, where the latent factors will be output", required=True)
    parser.add_argument("--log", type=str, help="log file", required=True)
    args = parser.parse_args()
    numLatentComponents = args.latent
    inputDir = args.input
    regularizationParam = args.lmbda
    outputEntities = args.outputentities
    outputFactors = args.outputfactors
    logFile = args.log
    
    logging.basicConfig(filename=logFile, filemode='w', level=logging.DEBUG)
    _log = logging.getLogger('RESCAL') 
    
    dim = 0
    with open('./%s/entity-ids' % inputDir) as entityIds:
        for line in entityIds:
            dim += 1
    print 'The number of entities: %d' % dim          
    
    numSlices = 0
    numNonzeroTensorEntries = 0
    X = []
    for inputFile in os.listdir('./%s' % inputDir):
        if fnmatch.fnmatch(inputFile, '[0-9]*-rows'):
            numSlices += 1
            row = loadtxt('./%s/%s' % (inputDir, inputFile), dtype=np.int32)
            if row.size == 1: 
                row = np.atleast_1d(row)
            col = loadtxt('./%s/%s' % (inputDir, inputFile.replace("rows", "cols")), dtype=np.int32)
            if col.size == 1: 
                col = np.atleast_1d(col)
            Xi = coo_matrix((ones(row.size),(row,col)), shape=(dim,dim), dtype=np.uint8).tolil()
            numNonzeroTensorEntries += row.size
            X.append(Xi)
            
    print 'The number of tensor slices: %d' % numSlices
    print 'The number of non-zero values in the tensor: %d' % numNonzeroTensorEntries
    
    result = rescal(X, numLatentComponents, lmbda=regularizationParam)
    print 'Objective function value: %.30f' % result[2]
    print '# of iterations: %d' % result[3] 
    #print the matrix of latent embeddings and matrix of latent factors
    A = result[0]
    savetxt(outputEntities, A)
    R = result[1]
    with file(outputFactors, 'w') as outfile:
        for i in xrange(len(R)):
            savetxt(outfile, R[i])
    
