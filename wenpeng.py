#!/usr/bin/env python

import time
import threading
import sys

def Traverse(rootDir): 
    fileNo=0
    for lists in os.listdir(rootDir): 
        path = os.path.join(rootDir, lists) 
        #print path 
        if os.path.isdir(path): 
            Traverse(path) 
        elif os.path.isfile(path):
            file = open(path)
            #ReadFile(file)
            ReadFile(file, -1)
            
            fileNo+=1
            '''
            if fileNo > 0:
                return
            '''

def calc_froebius_norm(m):
    time.sleep(1) 
    return m

def calc_norm(m, i, norms):

    print >> sys.stderr, 'Starting thread', i

    norm = calc_froebius_norm(m)
    norms[i] = norm

def main():

    matrixes = [1, 2, 3, 4]

    norms = [0] * len(matrixes)

    threads = []
    for i, m in enumerate(matrixes):

        t = threading.Thread(target=calc_norm, args=(m, i, norms))
        t.start()
        threads.append(t)

    for thread in threads:
        t.join()

    print >> sys.stderr, norms

if __name__ == '__main__':

    main()
