# -*- coding: utf-8 -*- 




import os
import math
import numpy

tensor=[[[0 for col in range(1)] for row in range(1)] for slice in range(10)]
dimension=1
wordlist=[]
'''
dir='//nfs/data7/ldc-corpora2/LDC2012T21-extract-by-bruder/by_year'
#inputFile=open('//mounts/data/proj/wenpeng/MC/firstCorpus.txt', 'w')
count=0
size=0
tmpSize=0
list = os.listdir(dir)  #列出目录下的所有文件和目录
for line in list:
    filepath = os.path.join(dir,line)
    if os.path.isfile(filepath):   #如果filepath是文件，直接列出文件名
        count+=1
        print filepath
        size+=os.path.getsize (filepath)
        tmpSize=size/(pow(1024,3)*1.0)   
        if tmpSize<20.0:  #小于10G
            
            print '当前已...........'+str(tmpSize)+'G'
            outputfile=open(filepath, 'r')
            aString=outputfile.read()
            outputfile.close()
            inputFile.write(aString.lower())
        else:          
            inputFile.close()
            print '一共读取了'+str(count)+'个文件,总大小为'+str(tmpSize)+'G'
            exit()
'''

def ReadFile(file):
    for line in file:
        tokens=line.split(' ')
        wordPosition=-1;
        for token in tokens:
            wordPosition=wordPosition+1
            if token not in wordlist: # a new token
                # expand data structure
                wordlist.append(token)
                for slice in range(10):
                    tensor[slice]+=[[0]*dimension]
                    dimension+=1
                    for row in range(dimension):
                        tensor[slice][row]+=[0]
                #update the right context of its preceding words, and update its left context
            middle=wordlist.index(token)
            for i in range(1,6):
                if wordPosition-i >=0:
                    context=wordlist.index(tokens[wordPosition-i])
                    tensor[4+i][context][middle]+=1 # should start from 4+1=5 until 9
                    tensor[5-i][middle][context]+=1
             

def Traverse(rootDir): 
    for lists in os.listdir(rootDir): 
        path = os.path.join(rootDir, lists) 
        print path 
        if os.path.isdir(path): 
            Traverse(path) 
        elif os.path.isfile(path):
            file = open(path)
            ReadFile(file)
         
'''    
a=[[0 for col in range(3)] for row in range(2)]
a[0][0]=1
print a
# append a row
a=[[0]*3]+a
a.insert(1,[0]*3)
print a
# delete a row
del a[0]
print a
# append a col
for i in range(len(a)):
    a[i]=[0]+a[i]
    a[i].insert(1,2)
print a
# delete a col
for i in range(len(a)):
    del a[i][0]
print a
'''          
dic={'12': 'dic', '13':'my', '11':'wenpeng', 'haha':'daiguo' }
for key in dic:
    print key, dic[key]

