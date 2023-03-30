#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


parser = MMCIFParser()
structure = parser.get_structure("135l", "135l.cif")
dict_prot = MMCIF2Dict("135l.cif")


# In[3]:


# retrieve X,Y,Z coordinates of water molecules
x_coord = [[j for j in dict_prot["_atom_site.Cartn_x"]] for i in dict_prot["_atom_site.group_PDB"] if i == "HETATM"][0]
y_coord = [[j for j in dict_prot["_atom_site.Cartn_y"]] for i in dict_prot["_atom_site.group_PDB"] if i == "HETATM"][0]
z_coord = [[j for j in dict_prot["_atom_site.Cartn_z"]] for i in dict_prot["_atom_site.group_PDB"] if i == "HETATM"][0]
# it turns out, the outer list comprehension doesn't locate pointer at "HETATM"
x_coord


# In[4]:


x = [[j for j in dict_prot["_atom_site.Cartn_x"]] for i in dict_prot["_atom_site.group_PDB"] if str(dict_prot[i]) == "ATOM"]


# In[ ]:


# This doesn't work out, either. The returned list is empty.
x = [j for j in dict_prot if dict_prot["_atom_site.group_PDB"] == "HETATM"]


# In[5]:


#obtain X,Y,Z coordinate of each amino acid and water molecule from crystallographic data set.
x_coord_h2o = []
y_coord_h2o = []
z_coord_h2o = []
x_coord_aa = []
y_coord_aa = []
z_coord_aa = []
ca_id = []

for i,j in enumerate(dict_prot["_atom_site.group_PDB"]):
    if j == "HETATM":
        x_coord_h2o.append(dict_prot["_atom_site.Cartn_x"][i])
        y_coord_h2o.append(dict_prot["_atom_site.Cartn_y"][i])
        z_coord_h2o.append(dict_prot["_atom_site.Cartn_z"][i])
    else:
        x_coord_aa.append(dict_prot["_atom_site.Cartn_x"][i])
        y_coord_aa.append(dict_prot["_atom_site.Cartn_y"][i])
        z_coord_aa.append(dict_prot["_atom_site.Cartn_z"][i])

for i, j in enumerate(dict_prot["_atom_site.label_atom_id"]):
    if j == "CA":
        ca_id.append(i)
        
coord_h2o = np.dstack((x_coord_h2o, y_coord_h2o, z_coord_h2o))
coord_aa = np.dstack((x_coord_aa, y_coord_aa, z_coord_aa))
coord_ca = coord_aa[0][ca_id]
h2o_id = dict_prot["_pdbx_nonpoly_scheme.pdb_seq_num"]


# In[6]:


for i in coord_h2o[0]:
    print(i[0])
    


# In[7]:


for j in coord_ca:
    print(j)


# In[8]:


# calculate rmsd of every water molecule to CA of each amino acid in the protein.
rmsd_ca =[]
for i in coord_h2o[0]:
    for j in coord_ca:
        rmsd_ca.append(((float(i[0]) - float(j[0]))**2 + (float(i[1]) - float(j[1]))**2 + (float(i[2]) - float(j[2]))**2) ** 0.5)

rmsd_ca = np.reshape(rmsd_ca, (int(coord_h2o.shape[1]), int(len(rmsd_ca)/coord_h2o.shape[1]))) 
# each row is a water molecule's distance to each CA postion of a residue in protein.
# no. of row = no. of water molecule; no. of column = no. of residues


# In[25]:


h2o_id


# In[10]:


coord_h2o.shape


# In[11]:


# max and min rmsd
rmsd_ca_max = np.amax(rmsd_ca, axis=1)
rmsd_ca_min = np.amin(rmsd_ca, axis=1)
# residue index corresponding max and min rmsd
rmsd_ca_max_index = np.argmax(rmsd_ca, axis=1)
rmsd_ca_min_index = np.argmin(rmsd_ca, axis=1)
# residue identity corresponding max and min rmsd
max_aa = np.array(dict_prot["_entity_poly_seq.mon_id"])[rmsd_ca_max_index]
min_aa = np.array(dict_prot["_entity_poly_seq.mon_id"])[rmsd_ca_min_index]


# In[12]:


rmsd_ca_max_index


# In[28]:


df_min_rmsd = pd.DataFrame(np.dstack(((h2o_id), min_aa, rmsd_ca_min_index + 1, rmsd_ca_min))[0], columns = ['resid', 'Residue', 'Resi index','MIN RMSD'])
df_max_rmsd = pd.DataFrame(np.dstack(((h2o_id), max_aa, rmsd_ca_max_index + 1, rmsd_ca_max))[0], columns = ['H2O index', 'Residue', 'Resi index','MAX RMSD'])

#save DataFrame to pickle file
df_min_rmsd.to_pickle("df_min_rmsd.pkl")
df_max_rmsd.to_pickle("df_max_rmsd.pkl")


# In[21]:


df_min_rmsd


# In[22]:


df_max_rmsd


# In[23]:


plt.plot(rmsd_ca_min)
#plt.plot(rmsd_ca_max)


# In[24]:


plt.scatter(rmsd_ca_min_index, rmsd_ca_min)
plt.scatter(rmsd_ca_max_index, rmsd_ca_max)
plt.xlabel("residue")
plt.ylabel('RMSD(anstrong)')
plt.legend(['min', 'max'])
plt.title('RMSD')


# In[ ]:



