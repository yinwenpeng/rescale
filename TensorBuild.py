# -*- coding: utf-8 -*- 

import collections
import os
import GlobalVariables as gl
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import eigsh
from numpy import dot, zeros, kron, array, eye, ones, savetxt, loadtxt
import ExtRescal.rescal as rescal
import logging
import threading
import cPickle as pickle
import math
import numpy as np


#if __name__ == '__main__': # this line is added by Thomas

def initialize(): # a list of dictionary
    #for i in range(gl.window*2):
    # we only consider 5 slices, from +1 to +5
    for i in range(gl.window):
        D=collections.defaultdict(int)
        gl.dictionaryList.append(D)
        

def ReadFile(file, limit ):
    print file
    #for line in file.readlines():
        
    for num_lines, line in enumerate(file):
        
        if limit > 0 and num_lines > limit:
            break
        
        line='<B> '+line[:len(line)-1]+' <B>'
        #print line
        #tokens=line.strip().lower().split(' ') # consider lowercases
        tokens=line.strip().split(' ')   # keep the original word forms
        
        indexes = []
        for token in tokens:            
            middle=gl.wordlist.get(token, -1)
            if middle == -1: # if is a new word
                gl.wordlist[token] = gl.dimension
                middle=gl.dimension
                gl.dimension+= 1 # gl.dimension count the current word amount
                
                if gl.dimension % 10000 == 0:
                    print gl.dimension
                
            indexes.append(middle) # 'indexes stores all the word index'
            #   
            #change 'wordPosition' to 'currentPosi'      
            currentPosi=len(indexes)-1   
            leftRange=gl.window if  currentPosi>= gl.window else   currentPosi      
            for i in range(1,leftRange+1):
                context=indexes[currentPosi-i]  # scan from right to left 
                '''
                gl.dictionaryList[gl.window-1+i][(context, middle)]+=1 #update slice 5th to 9th
                gl.dictionaryList[gl.window-i  ][(middle, context)]+=1  
                '''    
                gl.dictionaryList[i-1][(context, middle)]+=1 #update slice 0th to 4th

def Traverse(folders): 
    fileNo=0
    for rootDir in folders:
        if os.path.isfile(rootDir): #is file
            file = open(rootDir)
            ReadFile(file, -1)
            fileNo+=1
        else:          #is directory
            for lists in os.listdir(rootDir): 
                path = os.path.join(rootDir, lists) 
                list=[path] 
                Traverse(list) 


def eachSlice(slice, tensor):
    # extract by order
    print 'slice..', slice
    dict= sorted(gl.dictionaryList[slice].iteritems(), key=lambda d:d[0])
    rows=[]
    cols=[]
    data=[]
    for (row, col), value in dict:
        rows.append(row)
        cols.append(col)
        data.append(math.log(1+value/100.0)) # normalized values
    tensor[slice]=csr_matrix((data, (rows, cols)), shape=(gl.dimension, gl.dimension))


def traverseDictionaryList():
    #matrixes = [1, 2, 3, 4]

    tensor = [0] * len(gl.dictionaryList)

    threads = []
    for i, m in enumerate(gl.dictionaryList):

        t = threading.Thread(target=eachSlice, args=(i, tensor))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    return tensor

def formTensor():
    # extract by order
    print 'Building tensor....'       
    tensor=traverseDictionaryList()
    storeTensor(tensor)
    storeWordList()
    return
    #call rescal
    print 'Calling rescal...'
    logging.basicConfig(level=logging.INFO)
    #A, R, fit, itr, exectimes = rescal(tensor, 50, init='nvecs', lambda_A=10, lambda_R=10, compute_fit=True)
    A, R, fit, itr, exectimes = rescal.rescal(tensor, 100, lmbda=0)
    printWordEmbedding(A)
    printRelation(R)
    
'''
def formTensor():
    # extract by order
    print 'Building tensor....'    
    tensor=[]
    for slice in range(len(gl.dictionaryList)):
        print 'slice...'+str(slice)
        dict= sorted(gl.dictionaryList[slice].iteritems(), key=lambda d:d[0])
        rows=[]
        cols=[]
        data=[]
        for (row, col), value in dict:
            rows.append(row)
            cols.append(col)
            data.append(value)
        tensor.append(csr_matrix((data, (rows, cols)), shape=(gl.dimension, gl.dimension)))
    
    #call rescal
    print 'Calling rescal...'
    logging.basicConfig(level=logging.INFO)
    #A, R, fit, itr, exectimes = rescal(tensor, 50, init='nvecs', lambda_A=10, lambda_R=10, compute_fit=True)
    A, R, fit, itr, exectimes = rescal.rescal(tensor, 50, lmbda=0)
    printWordEmbedding(A)
    printRelation(R)
'''    


def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


def storeTensor(Tensor):
    for i in range(len(Tensor)):
        save_sparse_csr('/mounts/data/proj/wenpeng/Tensor/tensor/tensorSlice_'+str(i)+'_uppercase_log_plusRelation.npz', Tensor[i])       
    print 'Tensor is stored over!'
    '''
    with open('/mounts/data/proj/wenpeng/Tensor/tensor/tensorSlice_'+str(i)+'_uppercase_log_plusRelation.dat', 'wb') as outfile:
            pickle.dump(Tensor[i], outfile, pickle.HIGHEST_PROTOCOL)
    with open('/mounts/data/proj/wenpeng/Tensor/tensorSlice_0.dat', 'rb') as infile:
        x = pickle.load(infile)
    '''
def storeWordList():
    file=open('/mounts/data/proj/wenpeng/Tensor/tensor/wordlist_uppercase.txt', 'w')
    for word in gl.wordlist:
        file.write(word+' '+str(gl.wordlist[word])+'\n')
    file.close()
    print 'word list printed over!'
def printRelation(R):
    with file('/mounts/data/proj/wenpeng/Tensor/result/relations_giga_wiki_20140711.txt', 'w') as outfile:
        for i in xrange(len(R)):
            savetxt(outfile, R[i])
    print 'Relations are stored over!'

def printWordEmbedding(matrix):
    output= open('/mounts/data/proj/wenpeng/Tensor/result/word2embedding_giga_wiki_20140711.txt', 'w')
    index2word={}
    for word, index in gl.wordlist.items():
        index2word[index]=word
    for i in range(len(matrix)):
        output.write(index2word[i]+' ') #this means word printed according id
        for length in range(len(matrix[i])):
            output.write(str(matrix[i][length])+' ')
        output.write('\n')
    print 'Word embeddings are stored over!'



def cosVector(x,y):
    if(len(x)!=len(y)):
        print('error input,x and y is not in the same space')
        return;
    result1=0.0;
    result2=0.0;
    result3=0.0;
    for i in range(len(x)):
        result1+=x[i]*y[i]   #sum(X*Y)
        result2+=x[i]**2     #sum(X*X)
        result3+=y[i]**2     #sum(Y*Y)
    return result1/((result2*result3)**0.5) #结果显示

       
def findTopTenNeighbors():
    embeddingDict=loadEmbeddingFile('/mounts/data/proj/wenpeng/Tensor/result/word2embedding_calculus.txt')
    for word in ['microsoft','monday','batman','florida','dancing']:
        print 'Computing similarities for: '+ word
        embedding=embeddingDict[word]
        neighbor2simi={}
        count=0
        for neighbor in  embeddingDict.keys():
            count+=1
            if count%1000000 ==0:
                print count
            if cmp(neighbor, word) != 0:
                neighbor2simi[neighbor]=cosVector(embedding, embeddingDict[neighbor])
        dict= sorted(neighbor2simi.iteritems(), key=lambda d:d[1],reverse = True)
        for i in range(10):
            print dict[i]
            
            
        
    
def loadEmbeddingFile(embeddingFile):
    embeddingDict={}
    file = open(embeddingFile)
    for line in file:
        tokens=line[:len(line)-2].split(' ') # consider lowercases
        values=[]
        for i in range(1, len(tokens)):
            values.append(float(tokens[i]))
        embeddingDict[tokens[0]]=values
    print 'Embedding loading finished.'
    return embeddingDict
            
'''  
def PrintMatrix():
    output= open('matrix.txt', 'w')
    for row in range(len(gl.data)):
        output.write(str(gl.data[row])+' ')
    output.write('\n\n')
    for row in range(len(gl.rows)):
        output.write(str(gl.rows[row])+' ')
    output.write('\n\n')
    for row in range(len(gl.cols)):
        output.write(str(gl.cols[row])+' ')
    output.write('\n\n')
    print 'matrix printed over!'
    output.close()
'''

initialize()
folders=['/mounts/data/proj/wenpeng/PhraseEmbedding/enwiki-20130503-pages-articles-cleaned-tokenized','/nfs/data7/ldc-corpora2/LDC2012T21-extract-by-bruder/by_year']
#folders=['/mounts/data/proj/wenpeng/PhraseEmbedding/enwiki-20130503-pages-articles-cleaned-tokenized']
Traverse(folders)   
formTensor()

#findTopTenNeighbors()

