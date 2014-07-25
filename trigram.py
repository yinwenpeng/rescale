# -*- coding: utf-8 -*- 

outFile=open("text8", 'r')
inFile=open("trigram.txt",'w')
content=outFile.read()
split=' '
stringList=content.split(' ')
count=0
#print stringList[1]
#exit()
for i in range(3, len(stringList)):
    stringTmp=stringList[i-2]+'_'+stringList[i]
    inFile.write(stringTmp+' ')
print 'over'
outFile.close()
inFile.close()

