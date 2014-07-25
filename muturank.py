#!/usr/bin/env python
# -*- coding: utf-8 -*-
#主要用来生成system
import os
import decimal
import math
from math import sqrt
from math import exp
from math import pi
from math import log
import numpy as np

import numpy
from numpy import *
from scipy import *

import NMI   #这个NMI本来是导入的一个自己写的计算NMI值的文件,但是好像容易出错,最后使用的是五老师自己的计算NMI的工具,没用这个函数了

#这儿的full_filename文件应该是综合了那6个单独文件生成的,我当初将其命名为count.txt,就是每行对应一个researcher, 没列对应一个category,数值就是这个researcher在这个领域的publication数量.
def muturank(full_filename, categories, researchers, alpha, beta,k, C, para):   
    #Dir=os.path.abspath('..')      
    readPath=full_filename
    fin=open(readPath, 'r')
    content=fin.readlines()
    researchers=100   #这儿100只是为了使得读取数据量少,减少运行时间,方便每一次调试程序
    myList=[[0 for x in range(categories)] for y in range(researchers)]
    lineNo=0
    for line in content:
        entries=line.split(" ")#suppose space is used to seperate the entries
        for i in range(0, categories):
            myList[lineNo][i]=int(entries[i])#string->int
        lineNo=lineNo+1
        #print '读取count.txt第'+str(lineNo)+'行\n'
        if lineNo==researchers:
            break
    fin.close()
    #researchers=lineNo

    #下面抽取ground truth, 这个矩阵比myList多一列,最后一列用来保存学者pub最多的那个领域的编号,其余的值就是将myList归一化后的结果
    ground_truth=[[0.0 for x in range(categories+1)] for y in range(researchers)]
    for row in range(researchers):
        sum=0
        max=0
        index=0
        for col in range(categories):
            sum=sum+myList[row][col]
            if myList[row][col]>max:
                max=myList[row][col]
                index=col
        ground_truth[row][categories]=index
        for col in range(categories):
            ground_truth[row][col]=myList[row][col]/sum    
            
    print 'count.txt文件读取完毕....,下面构建高维矩阵...'
    #next, counstruct the multi-dimensional array

    myTensor=[[[0 for x in range(researchers)] for y in range(researchers)] for z in range(categories)]
    print 'mytensor空间申请完毕...'

    for z in range(categories):
        for x in range(researchers):
            for y in range(x+1, researchers):
                mean=(myList[x][z]+myList[y][z])/2.0
                if mean!=0:
                    #下面计算相似度的方法应该就是论文中实验部分给出的公式
                    similarity=math.exp(-((myList[x][z]-myList[y][z])/mean)**2)
                else:
                    similarity=0.0
                #print similarity
                myTensor[z][x][y]=similarity
                myTensor[z][y][x]=myTensor[z][x][y]#对称矩阵,且对角线元素为0
    print 'tensor三维矩阵构建完成....'

    #Next, the iterative steps of muturank
    #实验中,初始值和先验值都设为了1/n
    nodeScores_Init=[1.0/researchers for x in range(researchers)]
    nodeScores_Prior=nodeScores_Init
    nodeScores_Post=[0 for x in range(researchers)]   #装结果的
    cateScores_Init=[1.0/categories for z in range(categories)]
    cateScores_Prior=cateScores_Init
    cateScores_Post=[0 for z in range(categories)]    #装结果的
    #建立一个临时的跳转概率二维数组，不是对称的,最后一列保存总和
    nodePeerChooseProb=[[0 for x in range(researchers+1)] for y in range(researchers)] #节点选择节点的概率
    node_cateChooseProb=[[0 for z in range(categories+1)] for y in range(researchers)]   #节点选择关系的概率
    
    
    flag=0
    iterNo=1
    while flag!=1:#当还没收敛
        flag=1
        for x in range(researchers):#按行
            sum=0
            for y in range(researchers):#列
                if y!=x:#自己到自己不需要
                    prob=0
                    for z in range(categories):
                        prob=prob+myTensor[z][x][y]*cateScores_Prior[z]
                    nodePeerChooseProb[x][y]=prob
                    sum=sum+prob
            nodePeerChooseProb[x][researchers]=sum
    
        #建立一个node选关系的概率二维数组，不是对称的,最后一列保存总和
        #node_cateChooseProb=[[0 for z in range(categories+1)] for y in range(researchers)]
        for x in range(researchers):#按行
            sum=0
            for z in range(categories):#列
                prob=0
                for y in range(researchers):
                    prob=prob+myTensor[z][x][y]*cateScores_Prior[z]
                node_cateChooseProb[x][z]=prob
                sum=sum+prob
            node_cateChooseProb[x][categories]=sum
        #下面开始迭代
        #计算node值
        for x in range(researchers):
            sum=0
            for y in range(researchers):
                sum=sum+nodeScores_Prior[y]*(nodePeerChooseProb[y][x]/nodePeerChooseProb[y][researchers])
            sum=(1-alpha)*sum+alpha*nodeScores_Init[x]
            nodeScores_Post[x]=sum
            if math.fabs(nodeScores_Post[x]-nodeScores_Prior[x])>math.pow(10,-4):
                flag=0

        #计算cate值
        for z in range(categories):
            sum=0
            for x in range(researchers):
                sum=sum+nodeScores_Prior[x]*(node_cateChooseProb[x][z]/node_cateChooseProb[x][categories])
            sum=(1-beta)*sum+beta*cateScores_Init[z]
            cateScores_Post[z]=sum
            if math.fabs(cateScores_Post[z]-cateScores_Prior[z])>math.pow(10,-4):
                flag=0
        #覆盖原有nodeScores
        for x in range(researchers):
            nodeScores_Prior[x]=nodeScores_Post[x]
        for z in range(categories):
            cateScores_Prior[z]=cateScores_Post[z]
        print '完成第'+str(iterNo)+'轮迭代....flag='+str(flag)
        print nodeScores_Prior
        iterNo=iterNo+1
    print '恭喜，迭代收敛!!!' 
    #因为关系的分布已经得到,下面将多维的tensor合并为二维的相似性矩阵，输出myMatrix.txt文件中
    myMatrix=[[0 for x in range(researchers)] for y in range(researchers)]
    for x in range(researchers):
        for y in range(researchers):
            sum=0
            for z in range(categories):
                sum=sum+myTensor[z][x][y]*cateScores_Prior[z]
            myMatrix[x][y]=sum
    Dir=os.path.abspath('..') 
    fout=open(Dir+'/myMatrix.txt', 'w')
    for x in range(researchers):
        for y in range(researchers-1):
            fout.write(str(myMatrix[x][y])+' ')
        fout.write(str(myMatrix[x][y+1])+'\n')
    fout.close()
    print '多维关系图已经合并成功！！！下面开始spectral clustering...'
    SpectralClustering(researchers,k, myMatrix, C, para,categories, ground_truth)


#下面这个函数和上面那个功能应该相似,只是处理的是合成数据
def synthetic(folder, alpha, beta,k, C, para):   
    #Dir=os.path.abspath('..')      
    categories=4
    researchers=350
    '''
    ground_truth=[[0.0 for x in range(categories+1)] for y in range(researchers)]
    for i in range(50):
        ground_truth[i][categories]=0
        ground_truth[i][0]=1.0
    for i in range(50, 150):
        ground_truth[i][categories]=1
        ground_truth[i][1]=1.0
    for i in range(150, 350):
        ground_truth[i][categories]=2
        ground_truth[i][2]=1.0
     '''   
    ground_truth=[0 for x in range(researchers)]
    for i in range(50):
        ground_truth[i]=0
    for i in range(50,150):
        ground_truth[i]=1
    for i in range(150,350):
        ground_truth[i]=2
    myTensor=[[[0 for x in range(researchers)] for y in range(researchers)] for z in range(categories)]
    print 'mytensor空间申请完毕...'    
    
    filelist=os.listdir(folder)
    for file in filelist:      
        #文件名确定dimension
        dimension=int(file[0:1])
        filepath=os.path.join(folder,file)#文件名
        fin=open(filepath,'r')
        content=fin.readlines()
        lineNo=0
        for line in content:
            #print line
            entries=line.split(' ')#suppose space is used to seperate the entries
            for col in range(researchers):
                myTensor[dimension-1][lineNo][col]=float(entries[col])
            lineNo+=1
    
    print 'tensor三维矩阵构建完成....经验证，读取相似度文件没有错'
    
    #下面将其输出，按照cluto的格式需要
    for dimension in range(categories):
        fout=open(str(dimension+1)+'_simiFile.txt','w')
        fout.write(str(researchers)+' '+str(researchers**2)+'\n')
        for row in range(researchers):
            for col in range(researchers):
                if row==col:
                    fout.write(' '+str(col+1)+' '+str(1.0))
                else:
                    fout.write(' '+str(col+1)+' '+str(myTensor[dimension][row][col]))
            fout.write('\n')
    print '单个维度上的矩阵输出完毕'
    #exit(0)
                   
    #Next, the iterative steps of muturank
    nodeScores_Init=[1.0/researchers for x in range(researchers)]
    nodeScores_Prior=nodeScores_Init
    nodeScores_Post=[0 for x in range(researchers)]
    cateScores_Init=[1.0/categories for z in range(categories)]
    cateScores_Prior=cateScores_Init
    cateScores_Post=[0 for z in range(categories)]
    #建立一个临时的跳转概率二维数组，不是对称的,最后一列保存总和
    nodePeerChooseProb=[[0 for x in range(researchers+1)] for y in range(researchers)]
    node_cateChooseProb=[[0 for z in range(categories+1)] for y in range(researchers)]
    
    
    flag=0
    iterNo=1
    while flag!=1:#当还没收敛
        flag=1
        for x in range(researchers):#按行
            sum=0
            for y in range(researchers):#列
                if y!=x:#自己到自己不需要
                    prob=0
                    for z in range(categories):
                        prob=prob+myTensor[z][x][y]*cateScores_Prior[z]
                    nodePeerChooseProb[x][y]=prob
                    sum=sum+prob
            nodePeerChooseProb[x][researchers]=sum
    
        #建立一个node选关系的概率二维数组，不是对称的,最后一列保存总和
        #node_cateChooseProb=[[0 for z in range(categories+1)] for y in range(researchers)]
        for x in range(researchers):#按行
            sum=0
            for z in range(categories):#列
                prob=0
                for y in range(researchers):
                    prob=prob+myTensor[z][x][y]*cateScores_Prior[z]
                node_cateChooseProb[x][z]=prob
                sum=sum+prob
            node_cateChooseProb[x][categories]=sum
        #下面开始迭代
        #计算node值
        for x in range(researchers):
            sum=0
            for y in range(researchers):
                sum=sum+nodeScores_Prior[y]*(nodePeerChooseProb[y][x]/nodePeerChooseProb[y][researchers])
            sum=(1-alpha)*sum+alpha*nodeScores_Init[x]
            nodeScores_Post[x]=sum
            if math.fabs(nodeScores_Post[x]-nodeScores_Prior[x])>math.pow(10,-4):
                flag=0

        #计算cate值
        for z in range(categories):
            sum=0
            for x in range(researchers):
                sum=sum+nodeScores_Prior[x]*(node_cateChooseProb[x][z]/node_cateChooseProb[x][categories])
            sum=(1-beta)*sum+beta*cateScores_Init[z]
            cateScores_Post[z]=sum
            if math.fabs(cateScores_Post[z]-cateScores_Prior[z])>math.pow(10,-3):
                flag=0
                #print '某一个dimension的前后概率差的绝对值为'+str(math.fabs(cateScores_Post[z]-cateScores_Prior[z]))
        #覆盖原有nodeScores
        for x in range(researchers):
            nodeScores_Prior[x]=nodeScores_Post[x]
        for z in range(categories):
            cateScores_Prior[z]=cateScores_Post[z]
        print '完成第'+str(iterNo)+'轮迭代....flag='+str(flag)
        #print nodeScores_Prior
        iterNo=iterNo+1
    print '恭喜，迭代收敛!!!' 
    print '关系分布：'
    print cateScores_Prior
    #下面将多维的矩阵合并为二维的相似性矩阵，输出
    myMatrix=[[0 for x in range(researchers)] for y in range(researchers)]
    for x in range(researchers):
        for y in range(researchers):
            sum=0
            for z in range(categories):
                sum=sum+myTensor[z][x][y]*cateScores_Prior[z]
            myMatrix[x][y]=sum
    Dir=os.path.abspath('..') 
    fout=open(Dir+'/myMatrix.txt', 'w')
    fout.write(str(researchers)+' '+str(researchers**2)+'\n')
    for x in range(researchers):
        for y in range(researchers):
            if x==y:
                fout.write(' '+str(y+1)+' '+str(1.0))
            else:
                fout.write(' '+str(y+1)+' '+str(myMatrix[x][y]))
        fout.write('\n')
    fout.close()
    print '多维关系图已经合并成功！！！下面开始spectral clustering...'
    
    SpectralClustering(researchers,k, myMatrix, C, para,categories, ground_truth)



def SpectralClustering(researchers,K, myMatrix, C, para, categories, ground_truth):
    #第一步是计算啦普拉斯矩阵
    simiMatrix=np.zeros((researchers, researchers))
    for x in range(researchers):
        for y in range(researchers):
            simiMatrix[x][y]=myMatrix[x][y]

    UnitMatrix=np.eye(researchers)

    #度矩阵,注意已经是-1/2次方后的结果
    degreeMatrix=np.zeros((researchers, researchers))
    for x in range(researchers):
        degreeSum=0
        for y in range(researchers):
            degreeSum=degreeSum+simiMatrix[x][y]
        degreeMatrix[x][x]=degreeSum**(-0.5)

    lapMatrix=UnitMatrix-np.dot(np.dot(degreeMatrix,simiMatrix),degreeMatrix)

    evals, evecs=np.linalg.eig(lapMatrix)
    list=[]#装对应向量的大小   
    for x in range(len(evals)):
        list.append(evals[x])
    list.sort(reverse=False)
    #找对应的坐标
    matrixForGMM_NK=[[0 for x in range(K)]for y in range(researchers)]
    for i in range(K):#只需要前K个
        for j in range(len(evals)):
            if list[i]==evals[j]:
                for row in range(researchers):
                    matrixForGMM_NK[row][i]=evecs[j][row]

    #先求特征向量矩阵的行和
    for x in range(researchers):
        sum=0
        for y in range(K):
            sum=sum+matrixForGMM_NK[x][y]
        for y in range(K):
            matrixForGMM_NK[x][y]=matrixForGMM_NK[x][y]/sum
    #下面计算一下依据向量表示的点的相似度，看画出来的效果
    
    fout=open('simiOnVector.txt','w')
    for row in range(researchers):
        for col in range(researchers):
            sum1=0.0
            sum2=0.0
            sum3=0.0
            for dimension in range(K):
                #print matrixForGMM_NK[row][dimension],matrixForGMM_NK[col][dimension]
                sum1+=matrixForGMM_NK[row][dimension]**2
                sum2+=matrixForGMM_NK[col][dimension]**2
                sum3+=matrixForGMM_NK[row][dimension]*matrixForGMM_NK[col][dimension]
                #print sum1,sum2,sum3
            simi=(sum3)/(math.sqrt(sum1)*math.sqrt(sum2))
            if simi>0.4:
                fout.write(str(1)+' ')
            else:
                fout.write(str(0)+' ')
        fout.write('\n')
    print '根据向量表示的相似度矩阵已经保存到simiOnVector.txt文件里面'
    #exit(0)
    
    #下面利用cluto进行k-means聚类，先将输出到文件
    fout=open('inputFile_cluto.txt','w')
    fout.write(str(researchers)+' '+str(K)+'\n')
    for row in range(researchers):
        for col in range(K-1):
            fout.write(str(matrixForGMM_NK[row][col])+' ')
        fout.write(str(matrixForGMM_NK[row][K-1])+'\n')
    fout.close()
    print '输出到cluto文件结束...'
    #exit(0)
    os.system('./vcluster -clmethod=direct -colmodel=none inputFile_cluto.txt '+str(C))
    
    #下面是谱聚类调用的三种不同的基本聚类算法
    print 'k-means运行完毕，下面开始调用GMM_NK...'
    #kmeans(C, researchers, ground_truth)  #当时没有用我自己写的k-means算法,直接用的cluto工具聚类的
    
    newGMM(matrixForGMM_NK,C, researchers,K,para, categories, ground_truth)
    newGMM_NK(matrixForGMM_NK,simiMatrix, C, researchers,K,para, categories, ground_truth)

#下面的的k-means函数没有使用,直接用的cluto作为k-means算法的替代
def kmeans(C, researchers, ground_truth):
    probs=[[0 for x in range(C+1)] for y in range(researchers)]
    fin=open('inputFile_cluto.txt.clustering.'+str(C),'r')
    content=fin.readlines()
    lineIndex=0
    for line in content:
        #print int(line), researchers
        probs[lineIndex][int(line)]=1
        probs[lineIndex][C]=int(line)
        lineIndex=lineIndex+1
    fin.close()    
    #for x in range(researchers):
    #    print probs[x]
    #exit(0)
    cluster_label=[[0 for x in range(4)] for y in range(C)]
    for col in range(C):
        for row in range(researchers):
            if probs[row][col]==1:
                cluster_label[col][ground_truth[row]]+=1
        max=0
        index=0
        for i in range(3):
            if cluster_label[col][i]>max:
                max=cluster_label[col][i]
                index=i
        cluster_label[col][3]=index
    result_kmeans=[0 for x in range(researchers)]
    for x in range(researchers):
        result_kmeans[x]=cluster_label[probs[x][C]][3]
    print ground_truth
    print result_kmeans
    print 'ground_truth和k-means结果打印完毕'
    a=numpy.zeros((researchers,))
    b=numpy.zeros((researchers,))     
                 
    for x in range(researchers):
        a[x]=ground_truth[x]
        b[x]=result_kmeans[x]
            
    print 'NMI值为：'+str(NMI.nmi(a,b))    
    
#下面的GMM或者GMM_NK都是在cluto聚类结果的基础上进行再聚类的                    
def newGMM(matrixForGMM_NK, simiMatrix,C,researchers,K,para, categories, ground_truth):
    #首先对researchers进行随机类的初始化，方差和均值的确定
    kmeans_clusterOfObjects=[0 for x in range(researchers)]
    fin=open('inputFile_cluto.txt.clustering.'+str(C),'r')
    content=fin.readlines()
    lineIndex=0
    for line in content:
        kmeans_clusterOfObjects[lineIndex]=int(line)
        lineIndex+=1
    fin.close()
    #下面统计每个cluster的总元素数量
    eleCountForCluster=[0 for x in range(C)]
    for x in range(C):
        count=0
        for y in range(researchers):
            if kmeans_clusterOfObjects[y]==x:
                count+=1
        eleCountForCluster[x]=count

    #下面根据kmeans_objects_cluster里面的对象聚类结果计算每个类的均值,方差和先验
    means_vari_pri=[[0.0 for x in range(K+2)] for y in range(C)]
    #采取遍历kmeans_clusterOfObjects，每发现一个c类的就把它 向量加进去means_vari_pri，最后除以总数得到平均
    for x in range(researchers):
        for col in range(K):
            means_vari_pri[kmeans_clusterOfObjects[x]][col]+=matrixForGMM_NK[x][col]
    for row in range(C):
        for col in range(K):
            means_vari_pri[row][col]/=eleCountForCluster[row]
    #下面求方差，遍历kmeans_clusterOfObjects，吧对应的差加起来先放到means_vari_pri倒数第二列，最后平均
    for r in range(researchers):
        for col in range(K):
            means_vari_pri[kmeans_clusterOfObjects[r]][K]+=(matrixForGMM_NK[r][col]-means_vari_pri[kmeans_clusterOfObjects[r]][col])**2
    for row in range(C):
        means_vari_pri[row][K]/=eleCountForCluster[row]
    #下面计算cluster的先验
    for row in range(C):
        means_vari_pri[row][K+1]=(eleCountForCluster[row]*1.0)/researchers
    
    #下面进入循环，首先进行GMM概率计算
    likelihood=0
    flag=0
    iter=0
    probs_researchers_cluster=[[0.0 for x in range(C+1)] for y in range(researchers+1)]
    while(flag==0):
        flag=1
        for row in range(researchers):
            for col in range(C):
                variation=0.0
                for kth in range(K):
                    variation+=(matrixForGMM_NK[row][kth]-means_vari_pri[col][kth])**2
                prob=(1.0/sqrt(means_vari_pri[col][K]*math.pow(2*pi, K)))*math.exp(-(variation/(2*means_vari_pri[col][K])))
                probs_researchers_cluster[row][col]=prob

        #下面计算似然
        temp_likelihood=0
        for row in range(researchers):
            rowSum=0.0
            for col in range(C):
                rowSum+=probs_researchers_cluster[row][col]*means_vari_pri[col][K+1]
            temp_likelihood+=rowSum
        if temp_likelihood-likelihood>math.pow(10, -4):
            flag=0
            print '第'+str(iter+1)+'次迭代，temp_likelihood='+str(temp_likelihood)
            likelihood=temp_likelihood
            iter+=1
        else:
            print '恭喜，GMM已经收敛...,最大似然为:'+str(temp_likelihood)
            exit(0) #这儿好像应该改为返回而不是退出程序
        #下面结合先验，
        for row in range(researchers):
            rowSum=0.0
            for col in range(C):
                rowSum+=means_vari_pri[col][K+1]*probs_researchers_cluster[row][col]
            probs_researchers_cluster[row][C]=rowSum
            for col in range(C):
                probs_researchers_cluster[row][col]=(means_vari_pri[col][K+1]*probs_researchers_cluster[row][col])/probs_researchers_cluster[row][C]
        for col in range(C):
            colSum=0.0
            for row in range(researchers):
                colSum+=probs_researchers_cluster[row][col]
            probs_researchers_cluster[researchers][col]=colSum

        #下面计算均值，方差和先验,先把means_vari_pri矩阵清0
        for row in range(C):
            for col in range(K+2):
                means_vari_pri[row][col]=0.0
        for col in range(C):
            for row in range(researchers):
                for kth in range(K):
                    means_vari_pri[col][kth]+=probs_researchers_cluster[row][col]*matrixForGMM_NK[row][kth]
        for row in range(C):
            for col in range(K):
                means_vari_pri[row][col]=means_vari_pri[row][col]/probs_researchers_cluster[researchers][row]
        #下面求方差
        for row in range(C):
            sum=0.0
            for rth in range(researchers):
                variation=0.0
                for col in range(K):
                    variation+=(matrixForGMM_NK[rth][col]-means_vari_pri[row][col])**2    
                variation=variation*probs_researchers_cluster[rth][row]                  
                sum+=variation
            means_vari_pri[row][K]=sum/probs_researchers_cluster[researchers][row]
        
        #下面求先验
        for row in range(C):
            means_vari_pri[row][K+1]=probs_researchers_cluster[researchers][row]/researchers

def newGMM_NK(matrixForGMM_NK, simiMatrix,C,researchers,K,para, categories, ground_truth):
    #首先对researchers进行随机类的初始化，方差和均值的确定
    kmeans_clusterOfObjects=[0 for x in range(researchers)]
    fin=open('inputFile_cluto.txt.clustering.'+str(C),'r')
    content=fin.readlines()
    lineIndex=0
    for line in content:
        kmeans_clusterOfObjects[lineIndex]=int(line)
        lineIndex+=1
    fin.close()
    #下面统计每个cluster的总元素数量
    eleCountForCluster=[0 for x in range(C)]
    for x in range(C):
        count=0
        for y in range(researchers):
            if kmeans_clusterOfObjects[y]==x:
                count+=1
        eleCountForCluster[x]=count

    #下面根据kmeans_objects_cluster里面的对象聚类结果计算每个类的均值,方差和先验
    means_vari_pri=[[0.0 for x in range(K+2)] for y in range(C)]
    #采取遍历kmeans_clusterOfObjects，每发现一个c类的就把它 向量加进去means_vari_pri，最后除以总数得到平均
    for x in range(researchers):
        for col in range(K):
            means_vari_pri[kmeans_clusterOfObjects[x]][col]+=matrixForGMM_NK[x][col]
    for row in range(C):
        for col in range(K):
            means_vari_pri[row][col]/=eleCountForCluster[row]
    #下面求方差，遍历kmeans_clusterOfObjects，吧对应的差加起来先放到means_vari_pri倒数第二列，最后平均
    for r in range(researchers):
        for col in range(K):
            means_vari_pri[kmeans_clusterOfObjects[r]][K]+=(matrixForGMM_NK[r][col]-means_vari_pri[kmeans_clusterOfObjects[r]][col])**2
    for row in range(C):
        means_vari_pri[row][K]/=eleCountForCluster[row]
    #下面计算cluster的先验
    for row in range(C):
        means_vari_pri[row][K+1]=(eleCountForCluster[row]*1.0)/researchers
    
    #下面进入循环，首先进行GMM概率计算
    likelihood=0
    flag=0
    iter=0
    probs_researchers_cluster=[[0.0 for x in range(C+1)] for y in range(researchers+1)]
    probs_combineSimi=[[0.0 for x in range(C)] for y in range(researchers)]
    while(flag==0):
        flag=1
        #下面首先计算初始的GMM概率
        for row in range(researchers):
            for col in range(C):
                variation=0.0
                for kth in range(K):
                    variation+=(matrixForGMM_NK[row][kth]-means_vari_pri[col][kth])**2
                prob=(1.0/sqrt(means_vari_pri[col][K]*math.pow(2*pi, K)))*math.exp(-(variation/(2*means_vari_pri[col][K])))
                probs_researchers_cluster[row][col]=prob
        #下面结合相似度
        for row in range(researchers):
            for col in range(C):
                sum=0.0
                simiSum=0.0
                for rth in range(researchers):
                    sum+=probs_researchers_cluster[row][col]*simiMatrix[row][rth]
                    simiSum+=simiMatrix[row][rth]
                sum/=simiSum
                probs_combineSimi[row][col]=para*probs_researchers_cluster[row][col]+(1-para)*sum
        for row in range(researchers):
            for col in range(C):
                probs_researchers_cluster[row][col]=probs_combineSimi[row][col]
        #下面计算似然
        temp_likelihood=0
        for row in range(researchers):
            rowSum=0.0
            for col in range(C):
                rowSum+=probs_researchers_cluster[row][col]*means_vari_pri[col][K+1]
            temp_likelihood+=rowSum
        if temp_likelihood-likelihood>math.pow(10, -4):
            flag=0
            print '第'+str(iter+1)+'次迭代，temp_likelihood='+str(temp_likelihood)
            likelihood=temp_likelihood
            iter+=1
        else:
            print '恭喜，GMM_NK已经收敛...,最大似然为:'+str(temp_likelihood)   
            compute_lable_researchers(probs_researchers_cluster, means_vari_pri,researchers,K,C,ground_truth)             
            exit(0)
        #下面结合先验，
        for row in range(researchers):
            rowSum=0.0
            for col in range(C):
                rowSum+=means_vari_pri[col][K+1]*probs_researchers_cluster[row][col]
            probs_researchers_cluster[row][C]=rowSum
            for col in range(C):
                probs_researchers_cluster[row][col]=(means_vari_pri[col][K+1]*probs_researchers_cluster[row][col])/probs_researchers_cluster[row][C]
        for col in range(C):
            colSum=0.0
            for row in range(researchers):
                colSum+=probs_researchers_cluster[row][col]
            probs_researchers_cluster[researchers][col]=colSum

        #下面计算均值，方差和先验,先把means_vari_pri矩阵清0
        for row in range(C):
            for col in range(K+2):
                means_vari_pri[row][col]=0.0
        for col in range(C):
            for row in range(researchers):
                for kth in range(K):
                    means_vari_pri[col][kth]+=probs_researchers_cluster[row][col]*matrixForGMM_NK[row][kth]
        for row in range(C):
            for col in range(K):
                means_vari_pri[row][col]=means_vari_pri[row][col]/probs_researchers_cluster[researchers][row]
        #下面求方差
        for row in range(C):
            sum=0.0
            for rth in range(researchers):
                variation=0.0
                for col in range(K):
                    variation+=(matrixForGMM_NK[rth][col]-means_vari_pri[row][col])**2    
                variation=variation*probs_researchers_cluster[rth][row]                  
                sum+=variation
            means_vari_pri[row][K]=sum/probs_researchers_cluster[researchers][row]
        
        #下面求先验
        for row in range(C):
            means_vari_pri[row][K+1]=probs_researchers_cluster[researchers][row]/researchers

#下面函数为GMM或者GMM_NK聚类后,计算每个researcher的聚类后的label的,最后和groundtruth一起计算NMI
#大体流程是先后计算P(cluster|researcher)和P(label|cluster)
def compute_lable_researchers(probs_researchers_cluster, means_vari_pri,researchers,K,C,ground_truth):
    #计算聚类后每个researcher的lable:p(label|researcher)
    temp=probs_researchers_cluster
    #先要获得P(cluster|objects)及结合先验后的概率
    for row in range(researchers):
        rowSum=0.0
        for col in range(C):
            rowSum+=means_vari_pri[col][K+1]*probs_researchers_cluster[row][col]
        probs_researchers_cluster[row][C]=rowSum
        for col in range(C):
            probs_researchers_cluster[row][col]=(means_vari_pri[col][K+1]*probs_researchers_cluster[row][col])/probs_researchers_cluster[row][C]
    #下面确定researcher的聚类的类
    for row in range(researchers):
        max=0.0
        index=0
        for col in range(C):
            if probs_researchers_cluster[row][col]>max:
                max=probs_researchers_cluster[row][col]
                index=col
        probs_researchers_cluster[row][C]=float(index)
    
    #下面计算p(label|cluster)
    #先关键是确定每个类C在不同的label上的比例
    labelProportionOfClu=[[0 for x in range(4)] for y in range(C)]
    for row in range(researchers):
        labelProportionOfClu[int(probs_researchers_cluster[row][C])][ground_truth[row]]+=1
    for row in range(C):
        max=0
        index=0
        for col in range(3):
            if labelProportionOfClu[row][col]>max:
                max=labelProportionOfClu[row][col]
                index=col
        labelProportionOfClu[row][3]=index
    result_GMM_NK=[0 for x in range(researchers)]
    for x in range(researchers):
        result_GMM_NK[x]=labelProportionOfClu[int(probs_researchers_cluster[x][C])][3]

    print ground_truth
    print result_GMM_NK
    print 'ground_truth和GMM_NK结果打印完毕'
    a=numpy.zeros((researchers,))
    b=numpy.zeros((researchers,))     
                 
    for x in range(researchers):
        a[x]=ground_truth[x]
        b[x]=result_GMM_NK[x]
            
    print 'NMI值为：'+str(NMI.nmi(a,b))    
 
 
 
 
 
 
 
 
 
 
 
 