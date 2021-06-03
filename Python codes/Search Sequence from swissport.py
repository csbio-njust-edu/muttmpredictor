
# coding: utf-8

# In[10]:


# 1.NCBIlist
import pandas as pd 
data = pd.read_excel('../data/total_TMProteins.xlsx',sheet_name='disease')
NCBIlist = data['ACC']

#2. Search protein sequence
#%%writefile searchpoly2.py
from Bio import ExPASy
from Bio import SeqIO

b=open('../data/1--fasta/TM_disease_fasta.txt','w')
a=open('../data/1--fasta/in2excel_disease_fasta.txt','w')

for i in range(len(NCBIlist)):
    with ExPASy.get_sprot_raw(NCBIlist[i]) as handle:
        IDtemp='del_1_'+str(i)
        seq_record = SeqIO.read(handle, "swiss")
        b.write('>'+IDtemp+'|'+seq_record.id+'|'+seq_record.name+'|'+str(len(seq_record))+'\n')
        b.write(str(seq_record.seq)+'\n')
        
        print('>'+IDtemp+'|'+seq_record.id+'|'+seq_record.name+'|'+str(len(seq_record))+'\n')
        print(str(seq_record.seq)+'\n')
        
        a.write(seq_record.id+',')
        a.write(str(seq_record.seq)+'\n')
        
b.close()
a.close()


# In[13]:


#0.Mark unique label for each protein
IDlist = open('../data/IDlist.txt','r')
linesID = IDlist.readlines()
IDlist.close()


# 1.NCBIlist
NCBIlist = open('../data/NCBIlist.txt','r')
lines = NCBIlist.readlines()
NCBIlist.close()

namelist = []
for i in range(len(lines)):
    temp = lines[i].rstrip()
    namelist.append(temp)


#2. search protein sequence
#%%writefile searchpoly2.py
from Bio import ExPASy
from Bio import SeqIO

b=open('../data/4--fasta/TM_protein_fasta.txt','w')
a=open('../data/insert2excel.txt','w')

for i in range(len(namelist)):
    with ExPASy.get_sprot_raw(namelist[i]) as handle:
        IDtemp=linesID[i].rstrip()
        seq_record = SeqIO.read(handle, "swiss")
        b.write('>'+IDtemp+'|'+seq_record.id+'|'+seq_record.name+'|'+str(len(seq_record))+'\n')
        b.write(str(seq_record.seq)+'\n')
        
        print('>'+IDtemp+'|'+seq_record.id+'|'+seq_record.name+'|'+str(len(seq_record))+'\n')
        print(str(seq_record.seq)+'\n')

        a.write(seq_record.id+',')
        a.write(str(seq_record.seq)+'\n')
        
b.close()
a.close()


# In[ ]:




