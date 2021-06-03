
# coding: utf-8

# In[59]:


import sys,os,string

from Bio import SeqIO
import urllib.request

amino_acids = ['A','C','D','E','F','G','H','I','K','L','M','N','P','R','Q','S','T','V','W','Y']

dict_31 = {
    "GLY": "G","ALA": "A","VAL": "V","ILE": "I","LEU": "L","PRO": "P",
    "SER": "S","THR": "T","CYS": "C","MET": "M","ASP": "D","ASN": "N",
    "GLU": "E","GLN": "Q","LYS": "K","ARG": "R","HIS": "H","PHE": "F",
    "TYR": "Y","TRP": "W"}

dict_13={}
for key in dict_31.keys():
    dict_13[dict_31[key]]=key

#提取 物理化学属性特征
def parse_physico_chemical_property_aaindex(fn):
    """
    Parses file with physico chemical properties.
    See example http://www.genome.jp/dbget-bin/www_bget?aaindex:FASG890101
    :param filename:
    :return: returns dict key = one letter name of an amino acid
                          value = physico chemical property

    # KUHL950101 Hydrophilicity scale (Kuhn et al., 1995)
    # MITS020101 Amphiphilicity index (Mitaku et al., 2002)
    # ZIMJ680102 Bulkiness (Zimmerman et al., 1968)
    # GRAR740102 Polarity (Grantham, 1974)
    # CHAM820101 Polarizability parameter (Charton-Charton, 1982)
    # ZIMJ680104 Isoelectric point (Zimmerman et al., 1968)
    # CHOC760101 Residue accessible surface area in tripeptide (Chothia, 1976)
    # FAUJ880109 Number of hydrogen bond donors (Fauchere et al., 1988)
    # KLEP840101 Net charge (Klein et al., 1984)
    # LEVM760105 Radius of gyration of side chain (Levitt, 1976)
    # CEDJ970103 Composition of amino acids in membrane proteins (percent) (Cedano et al.,  1997)
    # TAKK010101 Side-chain contribution to protein stability (kJ/mol) (Takano-Yutani, 2001)

    """
    # ？？？fn是什么？
    f = open(fn, 'r')
    ls = f.readlines()
    f.close()
    ls_aa = ls[-3].rstrip().split()[1:] # pair of aa
    ls_p1 = ls[-2].rstrip().split() # value for 1st aa
    ls_p2 = ls[-1].rstrip().split() # value for 2nd aa

    if len(ls_aa)!=len(ls_p1) or len(ls_aa)!=len(ls_p2):
        raise ValueError("Bad Parsing")

    prop={}
    for i in range(len(ls_aa)):
        aa1=ls_aa[i].split('/')[0]
        aa2=ls_aa[i].split('/')[1]
        p1 = float(ls_p1[i])
        p2 = float(ls_p2[i])

        prop[aa1]=p1
        prop[aa2]=p2

    return prop


#提取替代打分矩阵特征
def parse_substitution_matrix_aaindex(fn):

    '''
    # NGPC000101 Substitution matrix (PHAT) built from hydrophobic and transmembrane regions of the Blocks database (Ng et al., 2000)
    # MUET010101 Non-symmetric substitution matrix (SLIM) for detection of homologous transmembrane proteins (Mueller et al., 2001)
    # HENS920102 BLOSUM62 substitution matrix (Henikoff-Henikoff, 1992)
    '''

    f=open(fn, 'r')
    ls =f.readlines()
    f.close()

    substitution_matrix = {}

    i_start=0
    for i, s in enumerate(ls):
        if s[0]!='M':continue
        i_start=i
        break

    rows = ls[i_start].rstrip().split()[3].strip(',') # ARNDCQEGHILKMFPSTWYV
    cols = ls[i_start].rstrip().split()[6].strip(',') # ARNDCQEGHILKMFPSTWYV

    if rows!=cols:
        print (rows, cols)
        raise ValueError('Matrix must be squared')

    for i in range(0,len(rows)):
        data = ls[i_start + 1 + i].rstrip().split() #-5      -6       0       6      -7       1      12
        for j in range(0,i+1):
            pair_aa = rows[i]+cols[j]
            value = float(data[j])
            substitution_matrix[pair_aa]=value

    return substitution_matrix

#提取替换打分矩阵特征
def parse_substitution_matrix_slim(fn):

    f = open(fn, 'r')
    ls = f.readlines()[1:] # first line is a comment
    f.close()

    substitution_matrix = {}

    ls_aa = ls[0].rstrip().split('\t')[1:] # first el is 'aa/aa'
    for s in ls[1:]:
        data = s.rstrip().split('\t')
        aa1 = data[0]
        for i in range(0,len(ls_aa)):
           aa2 = ls_aa[i]
           value = float(data[i+1])
           substitution_matrix[aa1+aa2]=value

    return substitution_matrix

##将3字母AA改为1字母AA
def getAA(aa, code=1):
# this function convert amino acid into one letter upper case for code=1(e.g. A), or to three letter upper case (e.g. ALA) for code=3
    if code==1:
        if len(aa)==1 and aa.upper() in dict_13.keys(): return aa.upper()
        if len(aa)==3 and aa.upper() in dict_31.keys(): return dict_31[aa.upper()]
    elif code==3:
        if len(aa)==1 and aa.upper() in dict_13.keys(): return dict_13[aa.upper()]
        if len(aa)==3 and aa.upper() in dict_31.keys(): return aa.upper()

    print (aa)
    raise ValueError('Wrong aa code')
    exit(1)

    return 1

#获取物理化学属性
def getPhysicoChemicalProperties(dir='../resource/', ext='.txt'):
    # this function returns 6-tuple of tabulated properties for the amino acid
    properties = ['KUHL950101', 'MITS020101', 'ZIMJ680102', 'GRAR740102',
                  'CHAM820101', 'ZIMJ680104', 'CHOC760101', 'FAUJ880109',
                  'KLEP840101', 'LEVM760105', 'CEDJ970103', 'TAKK010101']


    physico_chemical_properties = []
    for prop in properties:
        fn = dir+prop+ext
        if not os.path.exists(fn):
            print (properties)
            print (dir, prop, ext, fn)
            raise ValueError('no such file')
        physico_chemical_properties.append(parse_physico_chemical_property_aaindex(fn))

    return physico_chemical_properties


#获取BlosumScore
def getBlosumScore(aa1, aa2):
    # this function returns substitution score from the Blosum matrix for aa1 to aa2 mutation

    global blosum62_matrix

    a1 = getAA(aa1)
    a2 = getAA(aa2)

    if a1+a2 in blosum62_matrix.keys():
        return blosum62_matrix[a1+a2]
    elif a2+a1 in blosum62_matrix.keys():
        return blosum62_matrix[a2+a1]
    else:
        print(a1, a2)
        raise ValueError('No corresponding keys in the Blosum matrix')

    return -1

#获取PhatScore
def getPhatScore(aa1, aa2):
    # this function returns substitution score from the PHAT matrix for aa1 to aa2 mutation

    global phat_matrix

    a1 = getAA(aa1)
    a2 = getAA(aa2)

    if a1+a2 in phat_matrix.keys():
        return phat_matrix[a1+a2]
    elif a2+a1 in phat_matrix.keys():
        return phat_matrix[a2+a1]
    else:
        print (a1, a2)
        raise ValueError('No corresponding keys in the PHAT matrix')

    return -1

#获取SlimScore
def getSlimScore(aa1, aa2):
    # this function returns substitution score from the SLIM matrix for aa1 to aa2 mutation

    global slim_matrix

    a1 = getAA(aa1)
    a2 = getAA(aa2)

    if a1+a2 in slim_matrix.keys():
        return slim_matrix[a1+a2]
    else:
        print (a1, a2)
        raise ValueError('No corresponding keys in the SLIM matrix')

    return -1

#
def getGeneralDescriptorsUniprot(acc):

    link = "http://www.uniprot.org/uniprot/"+acc+".xml"
#     handle = urllib.urlopen(link)
    handle = urllib.request.urlopen(link)
    record = SeqIO.read(handle, "uniprot-xml")
    seq_length = len(record)

    nTM=0
    nHelix = 0
    nStrand = 0
    nTurn = 0
    for f in record.features:
        if f.type=='transmembrane region': nTM+=1
        if f.type=='helix': nHelix+=1
        if f.type=='strand': nStrand+=1
        if f.type=='turn': nTurn+=1

    # probably it is only for structure
    if nHelix==0 and nStrand==0 and nTurn==0:
        nHelix=-100
        nStrand=-100
        nTurn=-100

    s_log=''
    if nTM==0:
        s_log="Warning! There is no transmembrane regions! %s" % (acc)


    descriptor=[seq_length, nTM, nHelix, nStrand, nTurn]
    return descriptor, s_log


def calculateSequenceBasedDescriptor(acc, aa_wt, aa_mut, pos):
    # this functions calculate features from the sequence-based-descriptors and combine them into the one vector

    global properties

    aa_wt = getAA(aa_wt)
    aa_mut = getAA(aa_mut)

    descriptor = []
    s_log=''
    
    for prop in properties: descriptor.append(prop[aa_wt])
    for prop in properties: descriptor.append(prop[aa_mut])
    descriptor.append(getBlosumScore(aa_wt, aa_mut))
    descriptor.append(getPhatScore(aa_wt,aa_mut))
    descriptor.append(getSlimScore(aa_wt,aa_mut))
    descriptor.append(getSlimScore(aa_mut,aa_wt))
    features, message = getGeneralDescriptorsUniprot(acc)
    s_log+=message
    for feature in features: descriptor.append(feature)
    
    return descriptor, s_log


if __name__=="__main__":

    blosum62_matrix = parse_substitution_matrix_aaindex('../resource/substitution_matrix_blosum62.txt')
    phat_matrix = parse_substitution_matrix_aaindex('../resource/substitution_matrix_phat.txt')
    slim_matrix = parse_substitution_matrix_slim('../resource/substitution_matrix_slim161.txt')
    properties = getPhysicoChemicalProperties()

    flag_check = False
    if flag_check==True:
        aa_wt = 'L'
        aa_mut = 'W'
        descriptor=calculateSequenceBasedDescriptor('P41145', aa_wt, aa_mut, 0)
        print (descriptor)


# In[60]:


aa_wt = 'A'
aa_mut = 'R'
descriptor=calculateSequenceBasedDescriptor('P41145', aa_wt, aa_mut, 0)
print (descriptor)


# In[61]:


type(descriptor[0])


# In[62]:


# for i in range(len(descriptor[0])):
#     tp = descriptor[0][i]
#     print(type(tp),i)


# In[63]:


import pandas as pd
from pandas import DataFrame
#1.打开文件
fdir='../data/0--original TM data/TMdataset.xlsx'
sheet_name='delneu'
pd = pd.read_excel(fdir, sheet_name)
Variation = pd['Variation'].tolist()
Name = pd['UNIPROT_ACC'].tolist()
IDlist = pd['ID'].tolist()

#2.计算每个样本的sequence_based features
features = open('../data/TM_sequence_features.txt','w')
for i in range(len(Variation)):
    QianAA = Variation[i][0]
    POS = Variation[i][1:-1]
    houAA = Variation[i][-1]
    
    nameTEMP = Name[i]
    aa_wt = QianAA
    aa_mut = houAA
    descriptor=calculateSequenceBasedDescriptor(nameTEMP, aa_wt, aa_mut, POS)
    
    Temp = IDlist[i]
    features.write(Temp+'\t')
    print(Temp)
    print(descriptor[0])
    for j in range(len(descriptor[0])): #descriptor[i]的类型为tuple
        tp = descriptor[0][j]
        
        features.write(str(tp)+'\t')
    features.write('\n')

print('sucessful!')
features.close()


# In[64]:


# type(Variation)  #list
# Variation[0]  #'A1035V'
# type(Variation[0]) # str
# QianAA = Variation[0][0]
# POS = Variation[0][1:-1]
# houAA = Variation[0][-1]
# print(QianAA ,POS,houAA)

