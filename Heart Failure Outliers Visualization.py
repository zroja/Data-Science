
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats, integrate


# In[2]:


data= pd.read_csv("C:Downloads/HF_app_data_features_de.csv")
data2=pd.read_csv("C:Downloads/HF_bp_data_de.csv")


# In[3]:


data.head()


# In[4]:


data.plot(kind = "scatter", x = "dias_gr_90", y = "sys_gr_140")


# In[5]:


data.plot(kind = "line", x = "dias_gr_90", y = "sys_gr_140")


# In[6]:


data.plot(kind = "bar", x = "dias_gr_90", y = "sys_gr_140")


# In[7]:


data.plot(kind = "pie", x = "dias_gr_90", y = "sys_gr_140")


# In[8]:


data.head(60)


# In[9]:


import pandas as pd 
import seaborn as sns

df = pd.read_csv('Downloads/HF_bp_data_de.csv')

dg = df[['systolic_BP','Diastolic_BP']]

#outcome_column = "PAT_ID_D"
df_0=df[df[outcome_column] == 0].drop([outcome_column], axis=1).reset_index(drop=True)
df_1=df[df[outcome_column] == 1].drop([outcome_column], axis=1).reset_index(drop=True)
df_2=df[df[outcome_column] == 2].drop([outcome_column], axis=1).reset_index(drop=True)        

for column in df:
        sns.distplot(df_0[column])
        sns.distplot(df_1[column])
        sns.distplot(df_2[column])


# In[10]:


data2=pd.read_csv("C:Downloads/HF_bp_data_de.csv")
ax = sns.regplot(x="Diastolic_BP", y= "systolic_BP", data = data2[data2["PAT_ID_D"]=="PAT_HF_13"])


# In[11]:


data2=pd.read_csv("C:Downloads/HF_bp_data_de.csv")
ax = sns.regplot(x="Diastolic_BP", y= "systolic_BP", data = data2)


# In[ ]:


#data2=pd.read_csv("C:Downloads/HF_bp_data_de.csv")
an = pd.read_csv('out.csv',index_col=0)

ax = sns.violinplot(x="TYPE", y="VALUE", data=an, inner = None)
ax = sns.swarmplot(x="TYPE", y="VALUE", data=an,
                  color = "white", edgecolor = "gray")


# In[ ]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

df = pd.read_csv('out.csv',index_col=0)
df = df.groupby(df.index)

#plot
with PdfPages('graphs.pdf')as pdf_page:
    for group in df:
        data = group[1]
        
        sns.set(style="ticks", color_codes=True)
        plot = sns.swarmplot(x="TYPE", y="VALUE", data=data)
        plot = plot.set_title(group[0])
        fig = plot.get_figure()
        pdf_page.savefig(fig)


# In[ ]:


al = pd.read_csv('out.csv', index_col =0)
af = sns.boxplot(x = "TYPE", y = "VALUE", data = al, whis = np.inf)
af = sns.swarmplot(x= "TYPE", y="VALUE", data = al, color = ".2")


# In[ ]:


from scipy import interp
from sklearn import datasets, neighbors
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import make_pipeline

print(_doc_)

LW=2
RANDOM_STATE =42

class DummySampler(object):
    def sample(self, x, y):
        return x,y
    def fit(self, x, y):
        return self
    def fit_sample(self, x, y):
        return self.sample(x,y)
    
    cv= StratifiedKFold(n_splits=3)
    
data = datasets.fetch


