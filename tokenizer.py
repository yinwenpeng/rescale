# -*- coding: utf-8 -*- 

from nltk.tokenize import TreebankWordTokenizer

def tokenizer():
    s = '''Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks.'''
    TreebankWordTokenizer().tokenize(s)

def readInitialCorpus():
    readFile=open("/mounts/Users/student/wenpeng/workspace/phraseEmbedding/MicrosoftParaphrase/msr_paraphrase_test.txt", 'r')
    writeFile=open("tokenized_test.txt", 'w')
    lineCount=0
    for line in readFile:
        lineCount+=1
        if lineCount>1:
            tokens=line.split("\t")
            writeFile.write(tokens[0]+"\t")
            for ele in TreebankWordTokenizer().tokenize(tokens[3]):
                writeFile.write(ele+" ")
            writeFile.write('\t')
            for ele in TreebankWordTokenizer().tokenize(tokens[4]):
                writeFile.write(ele+" ")
            writeFile.write('\n')
    readFile.close()
    writeFile.close()
    print 'over'

def perWordPerLine():
    readFile=open('/mounts/Users/student/wenpeng/workspace/phraseEmbedding/MicrosoftParaphrase/tokenized_msr/tokenized_train.txt','r')
    writeFile=open('/mounts/Users/student/wenpeng/workspace/phraseEmbedding/MicrosoftParaphrase/tokenized_msr/one-word-per-line_train.txt','w')
    for line in readFile:
        tokens=line.split('\t')
        for word in tokens[1].split(' \n'):
            writeFile.write(word+"\n")
        #writeFile.write('\n')
        for word in tokens[2].split(' \n'):
            writeFile.write(word+'\n')
        #writeFile.write('\n')
    readFile.close()
    writeFile.close()
    print 'over'

#readInitialCorpus()    
perWordPerLine()