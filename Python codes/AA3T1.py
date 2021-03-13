
# coding: utf-8

# In[ ]:


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

