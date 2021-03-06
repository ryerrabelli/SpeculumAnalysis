#!/usr/bin/env python
# coding: utf-8

# # Setup

# ## Pip install

# In[1]:


# Don't forget to restart runtime after installing

get_ipython().run_line_magic('pip', 'install "labelbox[data]" --quiet')
# %pip freeze
# %pip freeze | grep matplotlib  # get version


# ## Base imports
# 

# In[2]:


import os
import sys
print(sys.version)
import json
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

import skimage
import skimage.io
#from PIL import Image
import PIL
import PIL.Image
import requests

import labelbox
#from labelbox.data.annotation_types import Geometry

import IPython.display
import matplotlib
import matplotlib.pyplot as plt
import plotly
import plotly.express as px


# In[46]:



notebook_filename = requests.get("http://172.28.0.2:9000/api/sessions").json()[0]["name"]

# Avoids scroll-in-the-scroll in the entire Notebook
def resize_colab_cell():
  display(IPython.display.Javascript('google.colab.output.setIframeHeight(0, true, {maxHeight: 10000})'))
get_ipython().events.register('pre_run_cell', resize_colab_cell)


#@markdown ### func `def get_path_to_save(...):`
def get_path_to_save(plot_props:dict=None, file_prefix="", save_filename:str=None, save_in_subfolder:str=None, extension="jpg", dot=".", create_folder_if_necessary=True):
    """
    Code created myself (Rahul Yerrabelli)
    """
    replace_characters = {
        "$": "",
        "\\frac":"",
        "\\mathrm":"",
        "\\left(":"(",
        "\\right)":")",
        "\\left[":"[",
        "\\right]":"]",
        "\\": "",
        "/":"-",
        "{": "(",
        "}": ")",
        "<":"",
        ">":"",
        "?":"",
        "_":"",
        "^":"",
        "*":"",
        "!":"",
        ":":"-",
        "|":"-",
        ".":"_",
    }

    # define save_filename based on plot_props
    if save_filename is None:
        save_filename = "unnamed"

    #save_path = f"../outputs/{notebook_filename.split('.',1)[0]}"
    save_path = [
                 "outputs",
                f"{notebook_filename.split('.',1)[0]}",
                ]
    if save_in_subfolder is not None:
        if isinstance(save_in_subfolder, (list, tuple, set, np.ndarray) ):
            save_path.append(**save_in_subfolder)
        else:  # should be a string then
            save_path.append(save_in_subfolder)
    save_path = os.path.join(*save_path)

    if not os.path.exists(save_path) and create_folder_if_necessary:
        os.makedirs(save_path)
    return os.path.join(save_path, file_prefix+save_filename+dot+extension)
    #plt.savefig(os.path.join(save_path, save_filename+dot+extension))


# In[8]:


#@title ## Mount google drive and import my code

mountpoint_folder_name = "drive"  # can be anything, doesn't have to be "drive"
project_path_within_drive = "PythonProjects/SpeculumAnalysis" #@param {type:"string"}
#project_path_within_drive = "UIUC ECs/Rahul_Ashkhan_Projects/SpeculumProjects_Shared/Analysis" #@param {type:"string"}
project_path_full = os.path.join("/content/",mountpoint_folder_name,
                        "MyDrive",project_path_within_drive)

get_ipython().run_line_magic('cd', '{project_path_full}')


# In[6]:



try:
    import google.colab.drive
    import os, sys
    # Need to move out of google drive directory if going to remount
    get_ipython().run_line_magic('cd', '')
    # drive.mount documentation can be accessed via: drive.mount?
    #Signature: drive.mount(mountpoint, force_remount=False, timeout_ms=120000, use_metadata_server=False)
    google.colab.drive.mount(os.path.join("/content/",mountpoint_folder_name), force_remount=True)  # mounts to a folder called mountpoint_folder_name

    if project_path_full not in sys.path:
        pass
        #sys.path.insert(0,project_path_full)
    get_ipython().run_line_magic('cd', '{project_path_full}')
    
except ModuleNotFoundError:  # in case not run in Google colab
    import traceback
    traceback.print_exc()


# # Data

# ## Read in the collected/labeled data

# ### Labelbox

# #### Option 1: Read from labelbox

# ##### Set up labelbox connection
# Works with LabelBox api (https://labelbox.com/), which is the tool I used to label all the distances on the images.

# In[11]:


# Add your labelbox api key and project
# Labelbox API stored in separate file since it is specific for a labelbox 
#account and shouldn't be committed to git. Contact the 
# team (i.e. Rahul Yerrabelli) in order to access to the data on your own account.
with open("auth/LABELBOX_API_KEY.json", "r") as infile:
  json_data = json.load(infile)
API_KEY = json_data["API_KEY"]
del json_data   # delete sensitive info

PROJECT_ID = "cl2cept1u4ees0zbx6uan5kwa"
DATASET_ID_Glove = "cl2cerkwd5gtd0zcahfz98401"; DATASET_NAME_Glove = "SpeculumWithGlove"
DATASET_ID_Condom = "cl2hu1u8z019a0z823yl5f8gr"; DATASET_NAME_Condom = "SpeculumWithCondom"

client = labelbox.Client(api_key=API_KEY)
del API_KEY   # delete sensitive info
project = client.get_project(PROJECT_ID)
dataset_glove = client.get_dataset(DATASET_ID_Glove)
dataset_condom = client.get_dataset(DATASET_ID_Condom)
# Alternative way to get dataset
# dataset = next(client.get_datasets(where=(labelbox.Dataset.name == DATASET_NAME)))

# Below code is from labelbox tutorial
# Create a mapping for the colors
hex_to_rgb = lambda hex_color: tuple(
    int(hex_color[i + 1:i + 3], 16) for i in (0, 2, 4))
colors = {
    tool.name: hex_to_rgb(tool.color)
    for tool in labelbox.OntologyBuilder.from_project(project).tools
}


# #### Save mask images 

# In[94]:


# Export labels created in the selected date range as a json-type (list of dicts/elements/lists):
labels2 = project.export_labels(download = True, start="2022-04-01", end="2022-06-01")


# In[153]:


labels3 = [value.copy() for value in labels2 ] # copy at least one level deep


# In[154]:


for ind in range(len(labels3)):
    # Simplify "Label" and "Reviews" by removing unnecessary variables and making the necessary ones at the top level
    # Thus, labels3 will be only 2 layers deep.
    if "Label" in labels3[ind]:
        coords = labels3[ind]["Label"]["objects"][0]["bbox"]
        for key, val in coords.items():
            labels3[ind]["Label"+"-"+key] = val
        # URL to download mask. Still has token in it
        labels3[ind]["Label_url"] = labels3[ind]["Label"]["objects"][0]["instanceURI"] 
        del labels3[ind]["Label"]

    # Remove special info ie emails, tokens (except the Label_url for now)
    labels3[ind].pop("Labeled Data", None)  # url with token in it
    labels3[ind].pop("View Label", None)  # url
    labels3[ind].pop("Created By", None)  # has email address
    labels3[ind].pop("Reviews", None)  # empty list


# In[142]:


# Download images from URLs
import urllib.request

for ind in range(len(labels3)):
    filename = labels3[ind]["External ID"].split(".jpg")[0] + "_label"
    filepath = get_path_to_save(save_filename=filename, extension="png")
    urllib.request.urlretrieve(labels3[ind]["Label_url"], filepath)


# In[155]:


# Remove Label_url as that URL has the Labelbox token in it
labelbox_df = pd.DataFrame.from_dict(labels3).set_index("External ID").drop(columns=["Label_url"])


# In[156]:


labelbox_df.to_csv( get_path_to_save(save_filename="labelbox_details", extension="csv") )


# ##### Get dataframe now that labelbox is set up

# In[ ]:


labels_df


# In[ ]:


image_labels = project.label_generator()
image_labels = image_labels.as_list()
labels_df = pd.DataFrame([[
                           label.data.external_id, 
                           label.annotations[0].value.end.x - label.annotations[0].value.start.x, 
                           label.annotations[0].value.end.y - label.annotations[0].value.start.y, 
                           label.annotations[0].value.start.x, 
                           label.annotations[0].value.start.y, 
                           label.data.url, 
                           label.uid
                           ] 
                          for label in image_labels],
                         columns=["Filename","x","y", "xstart","ystart","url", "Label ID"])
labels_df.to_csv("data/02_intermediate/labels_df"+".csv")
labels_df.to_pickle("data/02_intermediate/labels_df"+".pkl")

label_from_id_dict = {label.data.external_id: label for label in image_labels}
#with open("data/02_intermediate/label_from_id_dict"+".json", "w") as outfile:
#    json.dump(label_from_id_dict, outfile)   # Error: Object of type Label is not JSON serializable 


# #### Option 2: Read from labelbox csv if already saved there from previous run

# In[ ]:


labels_df = pd.read_csv("data/02_intermediate/labels_df.csv", index_col=0)
#with open("data/02_intermediate/label_from_id_dict"+".json", "r") as infile:
#    label_from_id_dict = json.load(infile)


# ### Read trial data from saved excel sheet

# In[ ]:


def handle_vertical_ht(x):
    if x=="BROKE":
        return 0
    elif type(x)==str and x.lower() in ["n/a","na","nan"]:
        return np.nan
    else:
        return float(x)

# Made Trial a str because it is not really being used as a numeric variable - better for plotting as it becomes a discrete variable instead of continuous (i.e. for color legend)
speculum_df_raw = pd.read_excel("data/01_raw/SpeculumTrialData.xlsx", index_col=0, sheet_name="AllTrials",
                                dtype={"Order": np.int32, "Spec Ang": np.int32, "Spec Ht": np.int32, 
                                       #"Vertical Height": np.float64, 
                                       "Trial": str, "Filename": str, "Speculum Type": str},
                                converters={"Vertical Height": handle_vertical_ht},
                                )    
#key_cols = ["Speculum Type","Spec Ang","Spec Ht","Size","Material","Material Type","Method"]
#speculum_df_raw.drop_duplicates(subset=key_cols).reset_index().drop("index",axis=1).reset_index().rename({"index":"Set"},axis=1)
#set_info = speculum_df_raw[key_cols].drop_duplicates().reset_index().rename({"index":"Set"},axis=1)
#speculum_df_raw_with_set = speculum_df_raw.merge(set_info, how="outer",on=key_cols)

speculum_df_notfailed = speculum_df_raw.dropna(axis="index", subset=["Filename"])   # Dropped the rows with failed trials

speculum_df_raw.to_csv("data/02_intermediate/speculum_df_raw"+".csv")
speculum_df_raw.to_pickle("data/02_intermediate/speculum_df_raw"+".pkl")
speculum_df_notfailed.to_csv("data/02_intermediate/speculum_df_notfailed"+".csv")
speculum_df_notfailed.to_pickle("data/02_intermediate/speculum_df_notfailed"+".pkl")


# ## Data rearranging

# ### Combine labelbox and excel sheet, calculate relative value

# In[ ]:


df_long=pd.merge(left=speculum_df_notfailed, right=labels_df, on="Filename")

glove_rows = df_long["Material Type"]=="Glove"
# The glove images got rotated 90 degrees. To fix this and clarify the directions of the opening, renaming the columns from x,y to wd and ht.
df_long.loc[ glove_rows,"wd"] = df_long.loc[ glove_rows].y
df_long.loc[ glove_rows,"ht"] = df_long.loc[ glove_rows].x
df_long.loc[ glove_rows,"wd_start"] = df_long.loc[ glove_rows].ystart
df_long.loc[ glove_rows,"ht_start"] = df_long.loc[ glove_rows].xstart

df_long.loc[~glove_rows,"wd"] = df_long.loc[~glove_rows].x
df_long.loc[~glove_rows,"ht"] = df_long.loc[~glove_rows].y
df_long.loc[~glove_rows,"wd_start"] = df_long.loc[~glove_rows].xstart
df_long.loc[~glove_rows,"ht_start"] = df_long.loc[~glove_rows].ystart
df_long = df_long.drop(columns=["x","y","xstart","ystart"])

df_long.head()

# Calculate relative value by dividing by the 0mmHg value
base_mmHg = 0 # mmHg
for ind in df_long["Order"].unique():
    df_long.loc[df_long["Order"]==ind,"wd_rel"]  = 1- df_long.loc[df_long["Order"]==ind].wd / df_long.loc[ (df_long["Order"]==ind) & (df_long["mmHg"]==base_mmHg) ].wd.item()
    df_long.loc[df_long["Order"]==ind,"ht_rel"]  = 1- df_long.loc[df_long["Order"]==ind].ht / df_long.loc[ (df_long["Order"]==ind) & (df_long["mmHg"]==base_mmHg) ].ht.item()
#df_long


# ### Get wide form

# In[ ]:


df_wide = df_long.pivot(index=
                        ["Order","Speculum Type","Size","Material","Material Type","Method","Spec Ang","Spec Ht","Trial","Vertical Height"], 
                        columns="mmHg", values=["wd_rel","ht_rel"]).reset_index("Vertical Height")
df_wide_flat = df_wide.copy()
df_wide_flat.columns = [".".join([str(item) for item in col]).strip(".") for col in df_wide_flat.columns.values]


# ### Order by set and the mmHg within that set (multiindex)

# In[ ]:


df_multiindex = df_long.set_index(["Order","mmHg"])
df_multiindex


# ### Save processed dfs

# In[ ]:


df_long.to_csv(  "data/03_processed/combined_df_long"+".csv")
df_long.to_excel("data/03_processed/combined_df_long"+".xlsx")
df_long.to_pickle("data/03_processed/combined_df_long"+".pkl")

df_wide.to_csv(  "data/03_processed/combined_df_wide"+".csv")
df_wide.to_excel("data/03_processed/combined_df_wide"+".xlsx")
df_wide.to_pickle("data/03_processed/combined_df_wide"+".pkl")

df_wide_flat.to_csv(  "data/03_processed/combined_df_wide_flat"+".csv")
df_wide_flat.to_excel("data/03_processed/combined_df_wide_flat"+".xlsx")
df_wide_flat.to_pickle("data/03_processed/combined_df_wide_flat"+".pkl")

df_multiindex.to_excel("data/03_processed/combined_df_multiindex"+".xlsx")   # assuming a multiindex wouldn't save well to a csv file
df_multiindex.to_pickle("data/03_processed/combined_df_multiindex"+".pkl")  


# ## Get aggregate df across trials

# In[ ]:


# Group by all the parameters that will be the same across different trials of the same object
consistent_cols = ["Speculum Type", "Spec Ang", "Spec Ht", "Size", "Material", "Material Type", "Method", "mmHg"]
aggregatable_cols = ["wd","ht","wd_rel","ht_rel", "Vertical Height"]
grouped_trials = df_long[consistent_cols+aggregatable_cols].groupby(consistent_cols)
#display(grouped_trials.describe())

def sem(x, ddof=1):   # ddof=1 to get sample standard deviation, not the population standard deviation (np's default)
    sem = np.std(x, ddof=ddof)/np.sqrt(len(x))

def nonnan(x):
    return x[~np.isnan(x)]

df_agg_long = grouped_trials.agg([np.mean, scipy.stats.sem, np.std, np.min, np.median, np.max, np.count_nonzero], ddof=1).reset_index()

df_agg_long_flat = df_agg_long.copy()
df_agg_long_flat.columns = [".".join(col).strip(".") for col in df_agg_long.columns.values]
#df_agg_long_flat

df_agg_long.to_csv(   "data/04_aggregated/combined_df_agg_long"+".csv")
df_agg_long.to_excel( "data/04_aggregated/combined_df_agg_long"+".xlsx")
df_agg_long.to_pickle("data/04_aggregated/combined_df_agg_long"+".pkl")
df_agg_long_flat.to_csv("data/04_aggregated/combined_df_agg_long_flat"+".csv")
df_agg_long_flat.to_pickle("data/04_aggregated/combined_df_agg_long_flat"+".pkl")


# # Skip ahead from loaded code

# In[ ]:


speculum_df_raw = pd.read_pickle("data/02_intermediate/speculum_df_raw"+".pkl")
speculum_df_notfailed = pd.read_pickle("data/02_intermediate/speculum_df_notfailed"+".pkl")

labels_df = pd.read_csv("data/02_intermediate/labels_df.csv", index_col=0)
#with open("data/02_intermediate/label_from_id_dict"+".json", "r") as infile:
#    label_from_id_dict = json.load(infile)
    
df_long = pd.read_pickle(  "data/03_processed/combined_df_long.pkl")
df_wide = pd.read_pickle(  "data/03_processed/combined_df_wide.pkl")
df_wide_flat = pd.read_pickle(  "data/03_processed/combined_df_wide_flat.pkl")

df_agg_long = pd.read_pickle("data/04_aggregated/combined_df_agg_long.pkl")
df_agg_long_flat = pd.read_pickle("data/04_aggregated/combined_df_agg_long_flat.pkl")

df_multiindex = pd.read_pickle("data/03_processed/combined_df_multiindex"+".pkl")

