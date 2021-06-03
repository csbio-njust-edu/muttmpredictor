
# coding: utf-8

# In[1]:


'''
---------------
  replace_char(string,char,index):用于前后AA的替换，string为sequence,char为突变后的AA，index为位置
  mut_split(string)：将类似'R127A'的信息分成：R,127,A--便于后面的使用
--------------
'''

##------字符串在python中是不可变数据类型，字符串转换列表替换并转换,实现替换-----------##
def replace_char(string,char,index):
     string = list(string)
     string[index] = char
     return ''.join(string)

##-------------读取突变前后、位置信息，并保存在qian,pos,hou中--------------##
def mut_split(string):
    qian = string[0]
    pos =  string[1:-1]
    hou =  string[-1]
    return (qian,pos,hou)


# In[6]:


"""
---------------
sequence_Modify(booksheet,sheet_name):
   将工作簿booksheet中的工作表sheet_name，按照突变信息，改写sequence
   并添加:label
   和 唯一标识:Name
---------------
"""
def sequence_Modify(booksheet,sheet_name): 
    import pandas as pd
    from pandas import DataFrame
    fdir='../data/'+booksheet
    
    pd = pd.read_excel(fdir, sheet_name=sheet_name)
    Name = pd['ID'].tolist()
    Variation = pd['Variation'].tolist()
    Sequence = pd['Sequence']##Sequence是Series类型，Sequence[0]是string类型

    # 修改mutation information
    fa = open("../data/4--fasta/TM_protein_fatsa.txt",'w')
    for i in range(len(Variation)):
        qian,pos,hou = mut_split(Variation[i])
        Sequence[i] = replace_char(Sequence[i],hou,int(pos)-1)
        
        #保存fasta文件
        fa.write('>'+Name[i]+'\n')
        fa.write(Sequence[i]+'\n')
    fa.close()
    print("save sucessful !")

## -----调用sequence_Modify-----##
sequence_Modify('../data/0--original TM data/TMdataset.xlsx','delneu')


