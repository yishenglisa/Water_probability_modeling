#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os.path
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from scipy.signal import argrelextrema
from scipy import stats
import sklearn
from sklearn.cluster import MeanShift
from sklearn.cluster import  estimate_bandwidth
from sklearn.model_selection  import LeaveOneOut
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
import itertools
#from DICT4A import DICT4A
#from DICT4A_ALLAT import DICT4A_ALLAT


# In[2]:


# Obtain KDE values from atomscore file for 135l
# KDE values are under "bfactor" column
# use os and glob to read multiple files in the same directory all at once

path = "135l_data"
kde_files = glob.glob(path + "/135l" + "*.csv")
li = []

for filename in kde_files:
    df = pd.read_csv(filename, index_col = None, header = 0)
    li.append(df)
band_135l = pd.concat(li, axis=0, ignore_index = True)


# In[3]:


# Obtain EDIA values for 135l

edia_135l = pd.read_csv('135l_data/EDIA_values_135l.csv')


# In[5]:


# Obtain information of each water molecule's closest neighbor.
min_rmsd = pd.read_pickle("df_min_rmsd.pkl")
max_rmsd = pd.read_pickle("df_max_rmsd.pkl")
min_rmsd['resid'] = min_rmsd['resid'].astype('int64')


# In[6]:


# Merge EDIA and KDE dataframes for 135l
edia_band_135l = edia_135l.merge(band_135l, left_on = 'Substructure id', right_on = 'resid')
#min_rmsd['resid'] = edia_band_135l['Substructure id'].astype(int)
#edia_band_aa = edia_band_135l.merge(min_rmsd, left_on = 'resid', right_on = 'H2O index')


# In[11]:


edia_band_135l.band.unique()


# In[12]:


min_rmsd["resid"].unique()


# In[13]:


for i in edia_band_135l['band'].unique():
    band_data = edia_band_135l[edia_band_135l["band"] == i]
    plt.scatter(band_data["bfactor"], band_data["EDIA"])


# In[14]:


band_data = []
band = []

for i in edia_band_135l['band'].unique():
    band_data.append(edia_band_135l[edia_band_135l["band"] == i])
    band.append(i)
    
fig, axs = plt.subplots(10,1, figsize = (6, 55))
for i in range(0, len(band_data)):
    axs[i].scatter(band_data[i]["bfactor"], band_data[i]["EDIA"])
    axs[i].set_xlabel("KDE probability")
    axs[i].set_ylabel("EDIA")
    axs[i].set_title("band {}".format(band[i]))
plt.savefig('135l_KDE probability_vs_EDIA.pdf')


# In[10]:


band_data[0]['resid'].unique()


# In[11]:


# Obtain information of each water molecule's closest neighbor.
min_rmsd = pd.read_pickle("df_min_rmsd.pkl")
max_rmsd = pd.read_pickle("df_max_rmsd.pkl")
min_rmsd['resid'] = min_rmsd['resid'].astype('int64')
edia_kde_aa1 = pd.merge(band_data[0], min_rmsd, on = "resid")
edia_kde_aa1


# In[12]:


groups = edia_kde_aa1.groupby('Resi index')
for name, group in groups:
    plt.scatter(group.bfactor, group.EDIA)
plt.title("band 1.0")
plt.xlabel("KDE Probability")
plt.ylabel("EDIA")
plt.savefig("by aa.pdf")


# In[13]:


# plot boxplot of KDE, grouped by EDIA values.

edia1 = edia_kde_aa1[edia_kde_aa1['EDIA'] < 0.3]
edia3 = edia_kde_aa1[edia_kde_aa1['EDIA'] > 0.6]
edia2 = edia_kde_aa1[~edia_kde_aa1['EDIA'].isin(pd.concat([edia1, edia3])['EDIA'])]

fig = plt.figure(figsize =(6, 5))
ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot([edia1['bfactor'], edia2['bfactor'], edia3['bfactor']])
ax.set_xticklabels(['EDIA<0.3', '0.3≤EDIA≤0.6',
                    'EDIA>0.6'])
plt.ylabel('KDE probability')
plt.title('KDE boxplot, all aa')
plt.savefig('KDE boxplot all aa.pdf')


# In[14]:


# create separate plots for hydrophobic, poloar, positively charged, and negatively charged residues.
np_aa = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP']
polar_aa = ['SER', 'PRO', 'ASN', 'GLN', 'THR', 'CYS', 'TYR']
pos_aa = ['ARG', 'LYS', 'HIS']
neg_aa = ['GLU', 'ASP']

edia1_np = edia1[edia1['Residue'].isin(np_aa)]
edia2_np = edia2[edia2['Residue'].isin(np_aa)]
edia3_np = edia3[edia3['Residue'].isin(np_aa)]

edia1_polar = edia1[edia1['Residue'].isin(polar_aa)]
edia2_polar = edia2[edia2['Residue'].isin(polar_aa)]
edia3_polar = edia3[edia3['Residue'].isin(polar_aa)]

edia1_pos = edia1[edia1['Residue'].isin(pos_aa)]
edia2_pos = edia2[edia2['Residue'].isin(pos_aa)]
edia3_pos = edia3[edia3['Residue'].isin(pos_aa)]

edia1_neg = edia1[edia1['Residue'].isin(neg_aa)]
edia2_neg = edia2[edia2['Residue'].isin(neg_aa)]
edia3_neg = edia3[edia3['Residue'].isin(neg_aa)]


fig, axs = plt.subplots(1, 4, sharey=True, figsize = (27, 6))

axs[0].boxplot([edia1_np['bfactor'], edia2_np['bfactor'], edia3_np['bfactor']])
axs[1].boxplot([edia1_polar['bfactor'], edia2_polar['bfactor'], edia3_polar['bfactor']])
axs[2].boxplot([edia1_pos['bfactor'], edia2_pos['bfactor'], edia3_pos['bfactor']])
axs[3].boxplot([edia1_neg['bfactor'], edia2_neg['bfactor'], edia3_neg['bfactor']])
axs[0].set_ylabel('KDE probability')
axs[0].set_title('Hydrophobic residue')
axs[1].set_title('Polar residue')
axs[2].set_title('Positively charged residue')
axs[3].set_title('Negatively charged residue')
for i in range(4):
    axs[i].set_xticklabels(['EDIA<0.3', '0.3≤EDIA≤0.6','EDIA>0.6'])

fig.suptitle('KDE boxplot, grouped by aa type', fontsize = 15)
    
plt.savefig('boxplot by aa type.pdf')


# In[15]:


# divide KDE vs. EDIA plot into four quadrants, manually inspect the quadrant with low KDE but high EDIA - why we didn't predict a water molecule despite its high EDIA value?
groups = edia_kde_aa1.groupby('Resi index')
fig, ax = plt.subplots()
for name, group in groups:
    plt.scatter(group.bfactor, group.EDIA)
plt.title("band 1.0")
plt.xlabel("KDE Probability")
plt.ylabel("EDIA")

x_avg = edia_kde_aa1['bfactor'].mean()
y_avg = edia_kde_aa1['EDIA'].mean()
x_med = edia_kde_aa1['bfactor'].median()
y_med = edia_kde_aa1['EDIA'].median()


ax.axvline(x_avg, c='k', lw=1)
ax.axhline(y_avg, c='k', lw=1)
ax.axvline(x_med, c='r', lw=1)
ax.axhline(y_med, c='r', lw=1)


# In[16]:


low_kde = edia_kde_aa1[edia_kde_aa1['bfactor'] <= x_avg]
low_kde_hi_edia = low_kde[low_kde['EDIA'] >= y_avg]


# In[18]:


# high KDE value
plt.hist(edia_kde_aa1['bfactor'])
plt.xlabel('KDE probability')
plt.ylabel('Counts')


# In[25]:


high_kde = edia_kde_aa1[edia_kde_aa1['bfactor'] > 1.0 ]
high_kde.sort_values("bfactor", ascending = False)


# In[52]:


low_kde_hexh2o_id = [132, 156, 173, 175, 201, 203, 206, 225, 228, 230]
high_kde_hexh2o_id = [130, 131, 137, 147, 158, 163, 211]
low_kde_h2o = edia_kde_aa1[edia_kde_aa1['Substructure id'].isin(low_kde_hexh2o_id)]
helix_h2o = edia_kde_aa1[edia_kde_aa1['Substructure id'].isin(low_kde_hexh2o_id + high_kde_hexh2o_id)]


# In[53]:


plt.hist(helix_h2o['bfactor'])
plt.xlabel('KDE probability')
plt.ylabel('Counts')


# In[56]:


plt.scatter(helix_h2o['bfactor'], helix_h2o['EDIA'])
plt.xlabel("KDE Probability")
plt.ylabel("EDIA")
plt.title('Helix region')


# In[48]:


low_kde_h2o.sort_values("bfactor", ascending = False)


# In[45]:


plt.hist(low_kde_h2o['bfactor'])
plt.xlabel('KDE probability')
plt.ylabel('Counts')


# In[68]:


low_kde2 = edia_kde_aa1[edia_kde_aa1['bfactor'] <= 0.3]
low_kde2.sort_values('bfactor', ascending = False)
plt.scatter(low_kde2['bfactor'], low_kde2['EDIA'])
plt.xlabel("KDE Probability")
plt.ylabel("EDIA")


# In[73]:


len(low_kde2)
low_kde2.sort_values('bfactor', ascending = False).head(5)


# In[ ]:




