
# coding: utf-8

# In[76]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
get_ipython().run_line_magic('matplotlib', 'inline')

# ax = plt.figure(figsize=(30,5),dpi=300)
# plt.figure(24)
plt.rc('font',family='Times New Roman')###-----------------

ax1 = plt.subplot(111) 
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(True)
##------------------- ACC -------------------------##
ACC=(0.6364,0.5636)
ACCStd = (0.020, 0.020)
##------------------- precision -------------------------##
precision=(0.6190,0.5750)
precisionStd = (0.021, 0.021)
##------------------- recall -------------------------##
recall=(0.8667,0.7667)
recallStd = (0.021, 0.021)
##------------------- F1-------------------------##
F1=(0.7222,0.6571)
F1Std = (0.021, 0.021)
##------------------- MCC-------------------------##
MCC=(0.2657,0.0969)
MCCStd = (0.021, 0.021)



index = np.arange(1,12,6)
bar_width = 1
opacity = 0.7
error_config = {'ecolor': '0.5'}

y_major_locator=MultipleLocator(0.1)
ax1=plt.gca()
ax1.yaxis.set_major_locator(y_major_locator)
plt.ylim(0,1)
ax1.tick_params(labelsize=10)

ax1.set_ylim(0,1)

#patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
rects6 = ax1.bar(index + 2*bar_width, ACC, bar_width,
                alpha=opacity, color=(0.1, 0.2, 0.5),
                yerr=ACCStd, error_kw=error_config,
                label='accuracy',hatch='/')
rects7 = ax1.bar(index + 3*bar_width, precision, bar_width,
                alpha=opacity, color=(0.1, 0.3, 0.5),
                yerr=precisionStd, error_kw=error_config,
                label='precision',hatch='\\\\')
rects8 = ax1.bar(index + 4*bar_width, recall, bar_width,
                alpha=opacity, color=(0.1, 0.4, 0.5),
                yerr=recallStd, error_kw=error_config,
                label='recall',hatch='//')
rects9 = ax1.bar(index + 5*bar_width, F1, bar_width,
                alpha=opacity, color=(0.1, 0.5, 0.5),
                yerr=F1Std, error_kw=error_config,
                label='F1',hatch='-')
rects10 = ax1.bar(index + 6*bar_width, MCC, bar_width,
                alpha=opacity, color=(0.1, 0.6, 0.5),
                yerr=MCCStd, error_kw=error_config,
                label='MCC',hatch='\\')

plt.axhline(y=0.2657,c="grey", ls="-.", lw=0.8)
plt.axhline(y=0.5636,c="grey", ls="-.", lw=0.8)
ax1.set_ylabel('evaluation values',fontsize=10)
ax1.set_xticks(index + bar_width*4 )
ax1.set_xticklabels(('WAPSSM features', 'PSSM features'),fontsize=10)

for rect in rects6:
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., 1.03*height,
            '%0.4f' % (height),ha='center', va='bottom',fontsize=10)         
for rect in rects7:
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., 1.03*height,
            '%0.4f' % (height),ha='center', va='bottom',fontsize=10)  

for rect in rects8:
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., 1.03*height,
            '%0.4f' % (height),ha='center', va='bottom',fontsize=10)  
for rect in rects9:
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., 1.03*height,
            '%0.4f' % (height),ha='center', va='bottom',fontsize=10)  
for rect in rects10:
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., 1.03*height,
            '%0.4f' % (height),ha='center', va='bottom',fontsize=10)  
    
plt.grid('off')

# plt.legend(ncol =3,loc='best')
plt.legend(ncol =3,loc='upper right', bbox_to_anchor=(0.98, 1.01))

# plt.legend([rects6, rects7], ['MMP', 'PredictSNP'], loc = 'upper center') 
# plt.legend([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8], ['MMP', 'PRE'])
plt.rcParams['figure.figsize'] = (16, 14) 
plt.rcParams['savefig.dpi'] = 300 
plt.rcParams['figure.dpi'] = 300 
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
#                 wspace=0.2, hspace=0.05)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None)
# plt.legend(loc='best',frameon=False, ncol=2,fontsize=5) 
# ax.legend(['MMP', 'PredictSNP'],loc='upper center',frameon=False, ncol=2,fontsize=5)
plt.savefig('./python figures/1--WAPSSM_PSSM_compare.tif', dpi=300,bbox_inches ='tight') 
plt.show()




import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
get_ipython().run_line_magic('matplotlib', 'inline')


ax = plt.figure(figsize=(30,5),dpi=300)
plt.figure(23)
plt.rc('font',family='Times New Roman')###-----------------

# ---------------------- Accuracy   -------------------------------##
ax1 = plt.subplot(231) 
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(True)
##------------------- WAPSSM -------------------------##
WAPSSM=(0.6364)
WAPSSMStd = (0.020)
##------------------- PSSM -------------------------##
PSSM=(0.5636)
PSSMStd = (0.018)
index = np.arange(1,3,2)
bar_width = 0.1
opacity = 0.7
error_config = {'ecolor': '0.5'}
#patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
rects6 = ax1.bar(index + 2*bar_width, WAPSSM, bar_width,
                alpha=opacity, color='dimgrey',
                yerr=WAPSSMStd, error_kw=error_config,
                label='WAPSSM',hatch='//')
rects7 = ax1.bar(index + 4*bar_width, PSSM, bar_width,
                alpha=opacity, color='black',
                yerr=PSSMStd, error_kw=error_config,
                label='PSSM',hatch='\\\\')
plt.axhline(y=0.6364,c="grey", ls="-.", lw=0.10)
# plt.axhline(y=0.8,c="grey", ls="-.", lw=0.8)
ax1.set_ylabel('ACC',fontsize=8,fontstyle='italic') ##--- fontstyle='italic'将字体倾斜

# ax1.set_xticks(1,3)
# ax1.set_xticklabels(('WAPSSM',' ','PSSM'),fontsize=5)
# ax1.set_xticks(index + bar_width*4)
# ax1.set_xticklabels(('WAPSSM','PSSM'),fontsize=5)

# y_major_locator=MultipleLocator(0.1)
# ax1=plt.gca()
# ax1.yaxis.set_major_locator(y_major_locator)
# plt.ylim(0,1)

ax1.set_ylim(0,1)
ax1.tick_params(labelsize=6)
for rect in rects6:
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., 1.03*height,
            '%0.4f' % (height),ha='center', va='bottom',fontsize=7)         
for rect in rects7:
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., 1.03*height,
            '%0.4f' % (height),ha='center', va='bottom',fontsize=7)  
plt.grid('off')


frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
ax1.set_xlabel('(A)',fontsize=5)



# ---------------------- recall  -------------------------------##
ax1 = plt.subplot(232) 
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(True)
##------------------- WAPSSM -------------------------##
WAPSSM=(0.6190)
WAPSSMStd = (0.020)
##------------------- PSSM -------------------------##
PSSM=(0.5750)
PSSMStd = (0.022)
index = np.arange(1,3,2)
bar_width = 0.1
opacity = 0.7
error_config = {'ecolor': '0.5'}
#patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
rects6 = ax1.bar(index + 2*bar_width, WAPSSM, bar_width,
                alpha=opacity, color='dimgrey',
                yerr=WAPSSMStd, error_kw=error_config,
                label='WAPSSM',hatch='//')
rects7 = ax1.bar(index + 4*bar_width, PSSM, bar_width,
                alpha=opacity, color='black',
                yerr=PSSMStd, error_kw=error_config,
                label='PSSM',hatch='\\\\')
plt.axhline(y=0.6190,c="grey", ls="-.", lw=0.5)
# plt.axhline(y=0.8,c="grey", ls="-.", lw=0.8)
ax1.set_ylabel('Recall',fontsize=8,fontstyle='italic') ##--- fontstyle='italic'将字体倾斜
# ax1.set_xticks(1,3)
# ax1.set_xticklabels(('WAPSSM',' ','PSSM'),fontsize=5)
# ax1.set_xticks(index + bar_width*4)
# ax1.set_xticklabels(('WAPSSM','PSSM'),fontsize=5)

# y_major_locator=MultipleLocator(0.1)
# ax1=plt.gca()
# ax1.yaxis.set_major_locator(y_major_locator)
# plt.ylim(0,1)
ax1.tick_params(labelsize=6)
## 设置Y轴的长度
ax1.set_ylim(0,1)
for rect in rects6:
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., 1.03*height,
            '%0.4f' % (height),ha='center', va='bottom',fontsize=7)         
for rect in rects7:
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., 1.06*height,
            '%0.4f' % (height),ha='center', va='bottom',fontsize=7)  
plt.grid('off')


frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)




# ---------------------- MCC -------------------------------##
ax1 = plt.subplot(233) 
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(True)
##------------------- WAPSSM -------------------------##
WAPSSM=(0.8667)
WAPSSMStd = (0.020)
##------------------- PSSM -------------------------##
PSSM=(0.7667)
PSSMStd = (0.018)
index = np.arange(1,3,2)
bar_width = 0.1
opacity = 0.7
error_config = {'ecolor': '0.5'}
#patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
rects6 = ax1.bar(index + 2*bar_width, WAPSSM, bar_width,
                alpha=opacity, color='dimgrey',
                yerr=WAPSSMStd, error_kw=error_config,
                label='WAPSSM',hatch='//')
rects7 = ax1.bar(index + 4*bar_width, PSSM, bar_width,
                alpha=opacity, color='black',
                yerr=PSSMStd, error_kw=error_config,
                label='PSSM',hatch='\\\\')
plt.axhline(y=0.8667,c="grey", ls="-.", lw=0.5)
# plt.axhline(y=0.8,c="grey", ls="-.", lw=0.8)
ax1.set_ylabel('MCC',fontsize=8,fontstyle='italic') ##--- fontstyle='italic'将字体倾斜
# ax1.set_xticks(1,3)
# ax1.set_xticklabels(('WAPSSM',' ','PSSM'),fontsize=5)
# ax1.set_xticks(index + bar_width*4)
# ax1.set_xticklabels(('WAPSSM','PSSM'),fontsize=5)

# y_major_locator=MultipleLocator(0.1)
# ax1=plt.gca()
# ax1.yaxis.set_major_locator(y_major_locator)
# plt.ylim(0,1)
ax1.tick_params(labelsize=6)
## 设置Y轴的长度
ax1.set_ylim(0,1)
for rect in rects6:
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., 1.03*height,
            '%0.4f' % (height),ha='center', va='bottom',fontsize=7)         
for rect in rects7:
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., 1.03*height,
            '%0.4f' % (height),ha='center', va='bottom',fontsize=7)  
plt.grid('off')


frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)



# ---------------------- Precision-------------------------------##
ax1 = plt.subplot(234) 
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(True)
##------------------- WAPSSM -------------------------##
WAPSSM=(0.7222)
WAPSSMStd = (0.020)
##------------------- PSSM -------------------------##
PSSM=(0.6571)
PSSMStd = (0.018)
index = np.arange(1,3,2)
bar_width = 0.1
opacity = 0.7
error_config = {'ecolor': '0.5'}
#patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
rects6 = ax1.bar(index + 2*bar_width, WAPSSM, bar_width,
                alpha=opacity, color='dimgrey',
                yerr=WAPSSMStd, error_kw=error_config,
                label='WAPSSM',hatch='//')
rects7 = ax1.bar(index + 4*bar_width, PSSM, bar_width,
                alpha=opacity, color='black',
                yerr=PSSMStd, error_kw=error_config,
                label='PSSM',hatch='\\\\')
plt.axhline(y=0.7222,c="grey", ls="-.", lw=0.5)
# plt.axhline(y=0.8,c="grey", ls="-.", lw=0.8)
ax1.set_ylabel('Pre',fontsize=8,fontstyle='italic') 
# ax1.set_xticks(1,3)
# ax1.set_xticklabels(('WAPSSM',' ','PSSM'),fontsize=5)
# ax1.set_xticks(index + bar_width*4)
# ax1.set_xticklabels(('WAPSSM','PSSM'),fontsize=5)

# y_major_locator=MultipleLocator(0.1)
# ax1=plt.gca()
# ax1.yaxis.set_major_locator(y_major_locator)
# plt.ylim(0,1)
ax1.tick_params(labelsize=6)
## 设置Y轴的长度
ax1.set_ylim(0,1)
for rect in rects6:
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., 1.03*height,
            '%0.4f' % (height),ha='center', va='bottom',fontsize=7)         
for rect in rects7:
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., 1.02*height,
            '%0.4f' % (height),ha='center', va='bottom',fontsize=7)  
plt.grid('off')


frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)



# ---------------------- F1---------- ---------------------##
ax1 = plt.subplot(235) 
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(True)
##------------------- WAPSSM -------------------------##
WAPSSM=(0.2657)
WAPSSMStd = (0.012)
##------------------- PSSM -------------------------##
PSSM=(0.0969)
PSSMStd = (0.015)
index = np.arange(1,3,2)
bar_width = 0.1
opacity = 0.7
error_config = {'ecolor': '0.5'}
#patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
rects6 = ax1.bar(index + 2*bar_width, WAPSSM, bar_width,
                alpha=opacity, color='dimgrey',
                yerr=WAPSSMStd, error_kw=error_config,
                label='WAPSSM',hatch='//')
rects7 = ax1.bar(index + 4*bar_width, PSSM, bar_width,
                alpha=opacity, color='black',
                yerr=PSSMStd, error_kw=error_config,
                label='PSSM',hatch='\\\\')
plt.axhline(y=0.2657,c="grey", ls="-.", lw=0.5)
# plt.axhline(y=0.8,c="grey", ls="-.", lw=0.8)
ax1.set_ylabel('$\mathregular{F_1}$',fontsize=8,fontstyle='italic')

# ax1.set_xticks(1,3)
# ax1.set_xticklabels(('WAPSSM',' ','PSSM'),fontsize=5)
# ax1.set_xticks(index + bar_width*4)
# ax1.set_xticklabels(('WAPSSM','PSSM'),fontsize=5)

# y_major_locator=MultipleLocator(0.1)
# ax1=plt.gca()
# ax1.yaxis.set_major_locator(y_major_locator)
# plt.ylim(0,1)
ax1.tick_params(labelsize=6)
## 设置Y轴的长度
ax1.set_ylim(0,1)
for rect in rects6:
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., 1.01*height,
            '%0.4f' % (height),ha='center', va='bottom',fontsize=7)         
for rect in rects7:
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., 1.01*height,
            '%0.4f' % (height),ha='center', va='bottom',fontsize=7)  
plt.grid('off')


frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)




# ---------------------- specificity---------- ---------------------##
ax1 = plt.subplot(236) 
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(True)
##------------------- WAPSSM -------------------------##
WAPSSM=(0.3600)
WAPSSMStd = (0.012)
##------------------- PSSM -------------------------##
PSSM=(0.3200)
PSSMStd = (0.022)
index = np.arange(1,3,2)
bar_width = 0.1
opacity = 0.7
error_config = {'ecolor': '0.5'}
#patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
rects6 = ax1.bar(index + 2*bar_width, WAPSSM, bar_width,
                alpha=opacity, color='dimgrey',
                yerr=WAPSSMStd, error_kw=error_config,
                label='WAPSSM',hatch='//')
rects7 = ax1.bar(index + 4*bar_width, PSSM, bar_width,
                alpha=opacity, color='black',
                yerr=PSSMStd, error_kw=error_config,
                label='PSSM',hatch='\\\\')
plt.axhline(y=0.3600,c="grey", ls="-.", lw=0.5)
# plt.axhline(y=0.8,c="grey", ls="-.", lw=0.8)
ax1.set_ylabel('Spe',fontsize=8,fontstyle='italic') 

# ax1.set_xticks(1,3)
# ax1.set_xticklabels(('WAPSSM',' ','PSSM'),fontsize=10)
# ax1.set_xticks(index + bar_width*4)
# ax1.set_xticklabels(('WAPSSM','PSSM'),fontsize=10)

# y_major_locator=MultipleLocator(0.1)
# ax1=plt.gca()
# ax1.yaxis.set_major_locator(y_major_locator)
# plt.ylim(0,1)
ax1.tick_params(labelsize=6)

ax1.set_ylim(0,1)
for rect in rects6:
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., 1.01*height,
            '%0.4f' % (height),ha='center', va='bottom',fontsize=7)         
for rect in rects7:
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., 1.07*height,
            '%0.4f' % (height),ha='center', va='bottom',fontsize=7)  
plt.grid('off')


frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)



# plt.legend([rects6, rects7], ['WAPSSM', 'PSSM'], bbox_to_anchor=(-0.09, 2.35),fontsize=7,frameon=False, ncol=2) 
# plt.legend([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8], ['WAPSSM', 'PSSM'])

# plt.legend([rects6, rects7], ['WAPSSM', 'PSSM'], bbox_to_anchor=(-0.09, 1.18),fontsize=7,frameon=False, ncol=2) 
plt.legend([rects6, rects7], ['WAPSSM', 'PSSM'], bbox_to_anchor=(1.1, 1.1),fontsize=7,frameon=False, ncol=1) 

plt.rcParams['figure.figsize'] = (16, 14) 
plt.rcParams['savefig.dpi'] = 300 
plt.rcParams['figure.dpi'] = 300
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.3, hspace=0.2)
# plt.legend(loc='best',frameon=False, ncol=2,fontsize=5) 
# plt.legend(['WAPSSM', 'PSSM'],loc='upper center',frameon=False, ncol=2,fontsize=5)


frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)

plt.savefig('./python figures/1--WAPSSM_PSSM_compare.tif', dpi=300,bbox_inches ='tight') 
plt.show()
