## Name: extractRSTDepMat.py
## Purpose: Extract RST Dependency matrix
## Author: Yangfeng Ji @ Comp Ling Lab, Gatech
## Date: 04-20-2013
## Time-stamp: <yangfeng 06/02/2013 14:39:51>

import transXml2Mat as x2m
import sys, os
import utilDiscourse as util

class DepMat:
    def __init__(self, fname, dp_dict, fname_boundary=None):
        """
        fname -
        dp_dict -
        fname_boundary -
        """
        self.fname = fname
        self.fname_boundary = fname_boundary
        self.dp_dict = dp_dict
        self.eduToken = ''
        self.eduMat = ''
        self.n_edu = 0


    def read(self, output, is_includetoken=True):
        f = open(self.fname, 'r')
        if self.fname_boundary is not None:
            f_boundary = open(self.fname_boundary, 'r')

        new_edu = []
        for idx, line in enumerate(f):
            line = line.replace('\n', '')
            if self.fname_boundary is None:
                line_list = line.split('\t')
                b_label = line_list[-1]
            else:
                b_label = f_boundary[idx].replace('\n', '')
            ## Check whether the EDU boundary
            if b_label == 'F': # If NOT
                new_edu.append(line)
            elif b_label == 'T': # if YES
                new_edu.append(line)
                self.n_edu += 1
                (dep_pair, edu_token) = self._processEDU(new_edu, output)
                #
                new_edu = []
                edu_mat = ''
                for key in dep_pair:
                    try:
                        pos = self.dp_dict[key]
                        edu_mat = edu_mat + str(pos) + ':1 '
                    except KeyError:
                        pass
                self.eduMat = self.eduMat + edu_mat + '\n'
                # Append edu_token into self.eduToken_list
                edu_token = ' '.join(edu_token)
                self.eduToken = self.eduToken + edu_token + '\n'
            else:
                raise ValueError('Unrecognized boundary label')
        return (self.eduMat, self.eduToken, self.n_edu)


    def _processEDU(self, line_list, output):
        """
        output:
        output = 1: only uni-gram
        output = 2: uni-gram and bi-gram
        output = 3: uni-gram, bi-gram and dep_rel

        Data format example:
        11	1	was	VBD	auxpass	12	F
        """
        token_dict = {0:'ROOT'}
        pos_dict = {0:'ROOT'}
        head_dict = {} # To be consistent with Wackypedia data
        deprel_dict = {}
        sent_token = []
        for item in line_list:
            item_list = item.split('\t')
            sent_token.append(item_list[2])
            if item_list[4] != 'None':
                try:
                    word_index = int(item_list[0])
                    token_dict[word_index] = item_list[2]
                    pos_dict[word_index] = item_list[3]
                    deprel_dict[word_index] = item_list[4]
                    head_dict[word_index] = int(item_list[5])
                except ValueError:
                    print item_list
                    sys.exit()
        depend_pair = []
        ## Construct dependency pair (actually, not just dependency pair,
        ## also includes uni-gram and bi-gram
        if output >= 1:
            for token in sent_token:
                depend_pair.append(('unigram', token))
        if output >= 2:
            for (idx, token) in enumerate(sent_token[:-1]):
                depend_pair.append(('bigram', token, sent_token[idx+1]))
        if output >= 3:
            for key, head_key in head_dict.items():
                try:
                    token = token_dict[key]
                    head_token = token_dict[head_key]
                    if output == 3:
                        depend_pair.append(('deppair', head_token, token))
                    elif output == 4:
                        dep_rel = deprel_dict[key]
                        depend_pair.append(('deppair', head_token, token, dep_rel.upper()))
                except KeyError:
                    pass
        return (depend_pair, sent_token)


def readFiles(output, fname_dp_dict, prefix, suffix, readPathList=['../Data'], writePath='../Data', debug=0):
    print 'Open files to write ...'
    prefix = os.path.join(writePath, prefix)
    fmat = open(prefix+'_matrix.txt', 'w')
    ftokens = open(prefix+'_tokens.txt', 'w')
    print 'Read dp_dict ...'
    fileMatIdx_dict = {}
    base_index = 0
    file_counter = 0
    dp_dict = x2m.readFile2Dict(fname_dp_dict)
    for read_path in readPathList:
        file_list = util.filterFiles(read_path, suffix)
        for fname in file_list:
            if debug > 0:
                print 'From {} read file {}'.format(read_path, fname)
            dm = DepMat(os.path.join(read_path, fname), dp_dict)
            eduMat, eduToken, n_edu = dm.read(output)
            fileMatIdx_dict.update({fname:(base_index, base_index+n_edu)})
            base_index += n_edu
            fmat.write(eduMat)
            ftokens.write(eduToken)
            # print 'eduMat =', eduMat
            # print 'eduToken =', eduToken
    fmat.close()
    ftokens.close()
    writeDict2File(prefix + '_file_mat_index.txt', fileMatIdx_dict)
    print 'Done'


def writeDict2File(fname, dict_inst):
    f = open(fname, 'w')
    for item in dict_inst.items():
        f.write(str(item))
        f.write('\n')
    f.close()

def main(output=1):
    # 'pos' | 'only-token' | 'token' | 'all'
    if output == 1:
        fname_dp_dict = './unigram-lemma_dict.txt'
        prefix = 'unigram-lemma'
    elif output == 2:
        fname_dp_dict = './bigram-lemma_dict.txt'
        prefix = 'bigram-lemma'
    elif output == 3:
        fname_dp_dict = './dep-lemma_dict.txt'
        prefix = 'dep-lemma'
    elif output == 4:
        fname_dp_dict = './dep_label-lemma_dict.txt'
        prefix = 'dep_label-lemma'
    suffix = '.lmfeat'
    readFiles(output, fname_dp_dict, prefix, suffix, debug=1)

if __name__ == '__main__':
    main(output=3)
