
# coding: utf-8

# In[13]:


def change3to1(x):
    if(x=='Ala'):
        x='A'
    elif(x=='Arg'):
        x='R'
    elif(x=='Asn'):
        x='N'
    elif(x=='Asp'):
        x='D'
    elif(x=='Cys'):
        x='C'
    elif(x=='Gln'):
        x='Q'
    elif(x=='Glu'):
        x='E'
    elif(x=='Gly'):
        x='G'
    elif(x=='His'):
        x='H'
    elif(x=='Ile'):
        x='I'
    elif(x=='Leu'):
        x='L'
    elif(x=='Lys'):
        x='K'
    elif(x=='Met'):
        x='M'
    elif(x=='Phe'):
        x='F'
    elif(x=='Pro'):
        x='P'
    elif(x=='Ser'):
        x='S'
    elif(x=='Thr'):
        x='T'
    elif(x=='Trp'):
        x='W'
    elif(x=='Tyr'):
        x='Y'
    elif(x=='Val'):
        x='V'
    else:
        x='error'
    return x


# In[14]:


yuanAA = open('../data/yuan3AA.txt','r')
AAlines = yuanAA.readlines()
yuanAA.close()

new1AA = open('../data/new3AA21.txt','w')
for i in range(len(AAlines)):
    temp = AAlines[i].rstrip()
    qianAA = temp[:3]
    pos = temp[3:-3]
    houAA = temp[-3:]
    

    newQAA = change3to1(qianAA)
    newHAA = change3to1(houAA)
    
    new = newQAA+'\t'+pos+'\t'+newHAA
#     new1AA.write(temp+','+new+'\n')
    new1AA.write(new+'\n')

print('sucessful')
new1AA.close()

