
# coding: utf-8

# In[5]:


import math
import os  
path = "../data/5--PSSM/logpssm"
files= os.listdir(path) 
filename=[]
for i in range(len(files)):
    tem=files[i].split('.')[0].upper()
    filename.append(tem)
    

ID =[] 
POS =[] 
label=[]  
IDread = open('../data/proteinID.txt','r')
try: 
    for line in IDread:
        temp=line.rstrip().split('\t')
        ID.append(temp[0])
        POS.append(temp[2])
        label.append(temp[1])
finally:
    IDread.close()

# features=open('./data/result/PSSM//BCSNP_pssm_features.txt','w')
a=open('../data/5--PSSM/PSSM7.txt','w') 

for j in range(len(filename)):
    try:
        xiabiao=ID.index(filename[j])
        Position= POS[xiabiao]
        f = open(path+"/"+files[j]) 
        flines=f.readlines() ## 
        
        num=int(Position)-1 ## 
        IDname=ID[xiabiao]
        labelname=label[xiabiao]
#         tempvalues=flines[num]
#         features.write(IDname+'\t'+labelname+'\t'+tempvalues)
        
        tempvalues=flines[num-3:num+4]
        a.write(IDname+'\t'+labelname+'\t')
        for k in range(len(tempvalues)):
            temp=tempvalues[k].lstrip().split('    ')[:-1]
            for i in range(len(temp)):
#                 print(float(temp[i]))
                if(math.isnan(float(temp[i]))):
                    print("值为空：",IDname )
                a.write(temp[i]+'\t')
        a.write('\n')
        
#         ###-------------   w=1    ---------###
#         tempvalues=flines[num] 
#         a.write(IDname+'\t'+labelname+'\t')
#         temp=tempvalues.rstrip()
#         a.write(temp+'\n')

    ##------  close file  -------------##
        f.close()
    ##------  exception  -------------##   
    except IndexError:
        print('please check!')
#         features.write('IndexError:please confirm!'+'\n')
# features.close()
a.close()


# In[12]:


def xpssm(pssmlist,f):
    new=[]
    for i in range(len(pssmlist)):
        new.append(pssmlist[i]*f)
    return new

import pandas as pd
a=open('../data/5--PSSM/WAPSSM7.txt','w') 
df1 = pd.read_csv('../data/5--PSSM/PSSM7.csv', encoding='gbk') 
ID = df1['ID'].tolist()
Label = df1['Label'].tolist()

for i in range(546): 
    a.write(str(ID[i])+'\t'+str(Label[i])+'\t')
        
#########--------- Information after the mutant site ------------####
    ##--------------------- -3  -------------##
    vf3=df1.loc[i,'pssm1':'pssm20'] 
    pf3=vf3.values.tolist()
    pssmf3=xpssm(pf3,0.04)
    ##--------------------- -2 -------------##
    vf2=df1.loc[i,'pssm21':'pssm40'] 
    pf2=vf2.values.tolist()
    pssmf2=xpssm(pf2,0.2673)
    ##--------------------- -1 -------------##
    vf1=df1.loc[i,'pssm41':'pssm60'] 
    pf1=vf1.values.tolist()
    pssmf1=xpssm(pf1,0.6827)
    

    for j in range(len(pssmf3)):## 
        a.write(str(pssmf3[j])+'\t')  
    for j in range(len(pssmf2)):## 
        a.write(str(pssmf2[j])+'\t')  
    for j in range(len(pssmf1)):## 
        a.write(str(pssmf1[j])+'\t')         
        
        
    ##---------- Information at the mutant site -------------##
    pssm20=[]
    mutdf=df1.loc[i,'pssm61':'pssm80']  
    pssm20=mutdf.values.tolist()  
    for j in range(len(pssm20)):
        a.write(str(pssm20[j])+'\t')
    
    
    #########--------- Information before the mutant site --------####
    ##--------------------- +1 -------------##
    vz1=df1.loc[i,'pssm81':'pssm100'] 
    pz1=vz1.values.tolist()
    pssmz1=xpssm(pz1,0.6827)
    ##--------------------- +2 -------------##
    vz2=df1.loc[i,'pssm101':'pssm120'] 
    pz2=vf2.values.tolist()
    pssmz2=xpssm(pz2,0.2673)
    ##--------------------- +3 -------------##
    vz3=df1.loc[i,'pssm121':'pssm140'] 
    pz3=vf3.values.tolist()
    pssmz3=xpssm(pz3,0.04)
    

    for j in range(len(pssmz3)):
        a.write(str(pssmz3[j])+'\t')    
    for j in range(len(pssmz2)):
        a.write(str(pssmz2[j])+'\t')    
    for j in range(len(pssmz1)):
        a.write(str(pssmz1[j])+'\t')    
        
    a.write('\n')
a.close()

