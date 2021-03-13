
# coding: utf-8

# In[ ]:


# coding: utf-8

# In[ ]:


"""
cut_seq(Sequence,pos,PSA_length):
Sequence:待截取的序列
pos：突变点的位置
PSA_length:最终需要的PSA长度，目前为了便于运算，设置为?AA

"""

##----以pos为中心，截取特定长度的AA----##
def cut_seq(Sequence,pos,PSA_length):
    posQ=Sequence[:int(pos)] ## 将进来的序列，以突变点为中心，分为前后两部分
    posH=Sequence[int(pos)+1:]
    
    ##-------字符型变量使用前要先定义，后使用-----##
    posQN=''
    posHN=''
    SequenceN=''
    tempQ=''
    tempH=''
    
    if(len(posQ)<int(0.5*PSA_length)  and len(posH)<int(0.5*PSA_length)):## pos前后都小于二分之一的PSA要求长度
        tempQ='X'*(int(0.5*PSA_length)-len(posQ))
        posQN=tempQ+posQ
        tempH='X'*(int(0.5*PSA_length)-len(posH)+1)
        posHN=posH +tempH 
        
    elif(len(posQ)>int(0.5*PSA_length)and len(posH)<int(0.5*PSA_length)):## pos前大于，后小于二分之一的PSA要求长度
        tempQ=posQ[-int(0.5*PSA_length):]
        posQN=tempQ
        tempH='X'*(int(0.5*PSA_length)-len(posH)+1)
        posHN=posH +tempH 
        
    elif(len(posQ)<int(0.5*PSA_length)and len(posH)>int(0.5*PSA_length)):## pos后大于，前小于二分之一的PSA要求长度
        tempQ='X'*(int(0.5*PSA_length)-len(posQ))
        posQN=tempQ+posQ
        tempH=posH[:int(0.5*PSA_length)+1]
        posHN = tempH 
        
    else:                                                               ## pos前后都等于二分之一的PSA要求长度
        tempQ=posQ[-int(0.5*PSA_length):]
        posQN=tempQ
        tempH=posH[:int(0.5*PSA_length)+1]
        posHN = tempH        
     
    SequenceN = posQN +posHN  ##  新的序列，用于返回
    return (SequenceN)


