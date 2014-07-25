# -*- coding: utf-8 -*- 

words={}
bigrams={}
pairs={}



sent2features={}
output= open('sentence2featureindex.txt', 'w') 

def extractTrainFile():
    trainFile = open("/mounts/Users/student/wenpeng/workspace/phraseEmbedding/MicrosoftParaphrase/one-word-per-line_train_lemmatized_tagged_parsed.txt")
    newSent=[]
    newDpIndex=[]
    sentCount=0
    for line in trainFile:
        line = line.strip()#remove the spaces at two ends
        if len(line)==0:
            sentCount+=1
            #if sentCount % 100 ==0:
            print 'Sentence: '+str(sentCount)
            addressNewSent(sentCount, newSent, newDpIndex)
            del newSent[:] 
            del newDpIndex[:] 
        else:
            tokens=line.split('\t')
            if tokens[3] not in words.keys():
                words[tokens[3]]=str(len(words)) # convert to str index
            newSent.append(tokens[3])
            newDpIndex.append(tokens[9])
    trainFile.close()
    output.close()


def addressNewSent(sentCount, newSent, newDpIndex):

    uni=[]
    bigram=[]
    pr=[]
    for i in range(len(newSent)):
        uni.append(words.get(newSent[i]))
        
        if i<len(newSent)-1:
            if newSent[i]+'--'+newSent[i+1] not in bigrams.keys():
                bigrams[(newSent[i], newSent[i+1])]=str(len(bigrams))
            bigram.append(bigrams.get((newSent[i], newSent[i+1])))
        
        pair=()
        if newDpIndex[i] is '0':
            pair=(newSent[i], '<root>')
        else:
            pair=(newSent[i], newSent[int(newDpIndex[i])-1])
        if pair not in pairs.keys():   
            pairs[pair]=str(len(pairs))
        pr.append(pairs.get(pair))

    sent2features[sentCount-1]=(uni, bigram, pr)
    '''
    #output.write(str(sentCount)+'\t')
    for word in newSent[:len(newSent)-1]:
        output.write(words.get(word)+' ')
    output.write(words.get(newSent[len(newSent)-1])+'\t')
    
    # bigram
    for i in range(len(newSent)-1):
        if newSent[i]+'--'+newSent[i+1] not in bigrams.keys():
            bigrams[newSent[i]+'--'+newSent[i+1]]=str(len(bigrams))
        if i<len(newSent)-1:
            output.write(bigrams.get(newSent[i]+'--'+newSent[i+1])+' ')
        else:
            output.write(bigrams.get(newSent[i]+'--'+newSent[i+1])+'\t')
    # pairs
    for i in range(len(newSent)):
        pair=''
        if newDpIndex[i] is '0':
            pair=newSent[i]+'--<root>'
        else:
            pair=newSent[i]+'--'+newSent[int(newDpIndex[i])-1]
        if pair not in pairs.keys():   
            pairs[pair]=str(len(pairs))
        
        if i==len(newSent)-1:
            output.write(pairs.get(pair)+' ')
        else:
            output.write(pairs.get(pair)+'\n')
    '''
            
if __name__ == '__main__':
    extractTrainFile() 

        
        
        