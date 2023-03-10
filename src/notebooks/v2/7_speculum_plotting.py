#!/usr/bin/env python
# coding: utf-8

# # Setup

# ## Pip install

# In[2]:


# Don't forget to restart runtime after installing

get_ipython().run_line_magic('pip', 'install "labelbox[data]" --quiet  # installs all required libraries plus extras required in manipulating annotations (shapely, geojson, numpy, PILLOW, opencv-python, etc.)')
get_ipython().run_line_magic('pip', 'install -U kaleido  --quiet # for saving the still figures')
get_ipython().run_line_magic('pip', 'install poppler-utils   # for exporting to .eps extension')
get_ipython().run_line_magic('pip', 'install plotly==5.7.0.    # need 5.7.0, not 5.5, so I can use ticklabelstep argument')

# %pip freeze
# %pip freeze | grep matplotlib  # get version


# ## Base imports
# 

# In[1]:


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


# In[2]:


#notebook_filename = requests.get("http://172.28.0.2:9000/api/sessions").json()[0]["name"]
notebook_filename = "7_speculum_plotting.ipynb"

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


# In[3]:


#@title ## Mount google drive and import my code

mountpoint_folder_name = "drive"  # can be anything, doesn't have to be "drive"
project_path_within_drive = "PythonProjects/SpeculumAnalysis" #@param {type:"string"}
#project_path_within_drive = "UIUC ECs/Rahul_Ashkhan_Projects/SpeculumProjects_Shared/Analysis" #@param {type:"string"}
project_path_full = os.path.join("/content/",mountpoint_folder_name,
                        "MyDrive",project_path_within_drive)

get_ipython().run_line_magic('cd', '{project_path_full}')


# In[4]:


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


# In[ ]:





# #### Option 1: Read from labelbox

# ##### Set up labelbox connection
# Works with LabelBox api (https://labelbox.com/), which is the tool I used to label all the distances on the images.

# In[5]:


# Add your labelbox api key and project
# Labelbox API stored in separate file since it is specific for a labelbox 
#account and shouldn't be committed to git. Contact the 
# team (i.e. Rahul Yerrabelli) in order to access to the data on your own account.
with open("auth/LABELBOX_API_KEY.json", "r") as infile:
  json_data = json.load(infile)
API_KEY = json_data["API_KEY"]
del json_data   # delete sensitive info

PROJECT_ID = "cl2cept1u4ees0zbx6uan5kwa"
DATASET_ID_Glove = "cl2cerkwd5gtd0zcahfz98401"; DATASET_NAME_Glove = "SpeculumDataset1_Glove"
DATASET_ID_Condom = "cl2hu1u8z019a0z823yl5f8gr"; DATASET_NAME_Condom = "SpeculumDataset1_Condom"
DATSET_ID_2_3 = "cleky2xtu19w3070qezkdbhd9"

client = labelbox.Client(api_key=API_KEY)
del API_KEY   # delete sensitive info
project = client.get_project(PROJECT_ID)
dataset_glove = client.get_dataset(DATASET_ID_Glove)
dataset_condom = client.get_dataset(DATASET_ID_Condom)
dataset_sterile = client.get_dataset(DATSET_ID_2_3)



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


# In[6]:


dataset = next(client.get_datasets(where=(labelbox.Dataset.name == "SpeculumDataset2_3")))


# In[19]:


for data_row in dataset_sterile.data_rows():
    print(data_row.external_id, data_row.created_at, data_row.updated_at, data_row.uid)
    break


# In[7]:


data_rows = dataset_sterile.data_rows()
data_df = pd.DataFrame([[
                data_row.external_id, data_row.created_at, data_row.updated_at, data_row.uid
             ]
             for data_row in data_rows],
             columns=["external_id","created_at","updated_at","uid"]
    )
data_df.sort_values(by="external_id")


# In[13]:


chicago_tz = pytz.timezone("America/Chicago") 

for data_row in dataset_sterile.data_rows():
    #data_row.update()
    ca = data_row.created_at
    print(type(data_row.created_at))
    print(data_row.external_id, data_row.created_at, data_row.updated_at, data_row.uid)
    datatime_str = data_row.external_id
    dt = datetime.datetime(int(datatime_str[0:4]),int(datatime_str[4:6]),int(datatime_str[6:8]),
                           int(datatime_str[9:11]),int(datatime_str[11:13]),int(datatime_str[13:15]),
                           tzinfo=chicago_tz)
    data_row.update(update_at=dt
               )
    break


# In[38]:


InvalidAttributeError: Field(s) ''createdAt'' not valid on DB type 'DataRow'("Field(s) ''createdAt'' not valid on DB type 'DataRow'", None)


# In[8]:


import datetime
import pytz

datatime_str = "20230225_173221.jpg"

chicago_tz = pytz.timezone("America/Chicago") 



# In[ ]:


datetime.datetime.now()


# In[1]:


datetime.datetime.strptime("20230225", '%y%m%d')


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

labels2 = project.export_labels(download = True, start="2022-04-01", end="2022-06-01")
labels3 = [value.copy() for value in labels2 ]


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


# # Set up for displaying

# In[ ]:


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

labels2 = project.export_labels(download = True, start="2022-04-01", end="2022-06-01")
labels3 = [value.copy() for value in labels2 ]


# In[ ]:


labels3


# ## Setup dicts and helper functions

# In[ ]:


category_orders={
    #"Size": ["S", "M", "L","Unspecified","None"],
    #"Size": ["S","Small", "M", "Medium", "L", "Large","Unspecified","None"],
    "Size": ["Small", "Medium", "Large","Unspecified","None", "S", "M", "L"],  # change the order of S vs Small etc changes the color
    "Material":["Nitrile","Vinyl","Trojan", "Lifestyle", "Durex", "Skyn","None"],
    "Material Type":["Glove","Condom","None"],
    "Method":["Middle","Two","Palm","Middle finger","Two fingers","Palm","Precut","None"],
    "Speculum Type":["Yellow","White","Green","Blue"]
    }
labels = {
    "Trial":            "<b>Trial #</b>",
    "wd_rel":           "<b>Relative Obstruction</b>",
    "wd_rel.mean":      "<b>Mean Relative Obstruction (S.E.)</b>", 
    "mmHg":             "<b>Pressure (mmHg)</b>", 
    "Material":         "<b>Material</b>", 
    "Material Type":    "<b>Material Type</b>",
    "Size":             "<b>Size</b>",
    "Method":           "<b>Method</b>",
    "Brand":            "<b>Brand</b>",
    "Day Ct":           "<b>Day Ct</b>",
    "Set Ct":           "<b>Set Ct</b>",
    "Day Set Ct":       "<b>Day Set Ct</b>",
}

color_discrete_map = {
    "Medium":           px.colors.qualitative.Safe[1],
    "Nitrile":          px.colors.qualitative.Safe[1],
    "Middle finger":    px.colors.qualitative.Safe[1],
    "Small":            px.colors.qualitative.Safe[0],
    "Large":            px.colors.qualitative.Safe[2],
    "Two fingers":      px.colors.qualitative.Safe[4],
    "Vinyl":            px.colors.qualitative.Safe[6],

    "Trojan":           px.colors.qualitative.Safe[7], 
    "Lifestyle":        px.colors.qualitative.Safe[3],
    "Durex":            px.colors.qualitative.Safe[9],
    "Skyn":             px.colors.qualitative.Safe[5],
    "None":             px.colors.qualitative.Safe[6],

    "Yellow": "yellow",
    "White": "grey",
    "Green": "green",
    "Blue": "blue",
}
# pattern shape options =  [ "", "/", "\\", "x", "-", "|", "+", "." ]
pattern_shape_map = {
    "Medium":           "+",
    "Nitrile":          "+",
    "Middle finger":    "+",
    "Small":            "/",
    "Large":            "\\",
    "Vinyl":            "x",
    "Two fingers":      "|",

    "Trojan":           "/", 
    "Lifestyle":        "-", 
    "Durex":            ".", 
    "Skyn":             "\\",
    "None":             "",
}


def criteria_to_str(criteria:dict) -> str:
    return ", ".join([f"{labels.get(key) or key}={val}".replace("<br>","").replace("<b>","").replace("</b>","") for key,val in criteria.items()])


def filter_by_criteria(criteria:dict, starting_df:pd.DataFrame) -> pd.DataFrame:
    #df_sampled = df_agg_long_flat.loc[ np.all([df_agg_long[arg]==val for arg, val in criteria.items()], axis=0) ]
    #df_sampled = df_agg_long_flat.loc[ np.all([ (type(val)!=list and df_agg_long[arg]==val ) or np.in1d(df_agg_long[arg],val)  for arg, val in criteria.items()], axis=0) ]
    #starting_df.loc[ np.all([ (type(val)!=list and starting_df[arg]==val ) or np.in1d(starting_df[arg],val)  for arg, val in criteria.items()], axis=0) ]
    conditions = []
    for arg, val in criteria.items():
        if hasattr(val,"__iter__") and not isinstance(val,str):
            conditions.append( np.in1d(starting_df[arg],val) )
        else:
            conditions.append( starting_df[arg]==val )
    return starting_df.loc[ np.all(conditions, axis=0) ]


# ## Setup plotly figure saving

# In[ ]:


default_plotly_save_scale = 4
def save_plotly_figure(fig, file_name:str, animated=False, scale=None, save_in_subfolder:str=None, extensions=None):
    """
    - for saving plotly.express figures only - not for matplotlib
    - fig is of type plotly.graph_objs._figure.Figure,
    - Requires kaleido installation for the static (non-animated) images
    """    
    if scale is None:
        scale = default_plotly_save_scale
    if extensions is None:
        extensions = ["html"]
        if not animated:
            # options = ['png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'eps', 'json']
            extensions += ["eps","png","pdf"]

    for extension in extensions:
        try:
            if extension in ["htm","html"]:
                #fig.update_layout(title=dict(visible=False))
                fig.write_html( get_path_to_save(save_filename=file_name, save_in_subfolder=save_in_subfolder, extension=extension), 
                    full_html=False,
                    include_plotlyjs="directory" )
            else:
                #if extension == "png":
                #    fig.update_layout(title=dict(visible=False))
                fig.write_image(get_path_to_save(save_filename=file_name, save_in_subfolder=save_in_subfolder, extension=extension), scale=scale)
        except ValueError as exc:
            import traceback
            #traceback.print_exception()

#col_options = {col_name:pd.unique(df_long[col_name]).tolist() for col_name in consistent_cols}
#display(col_options)


# ## Setup for plotting aggregates

# In[ ]:


def customize_figure(fig, width=640, height=360, by_mmHg=True, br_ct=1, space_ct=1, textposition="inside", textfont_color=None) -> dict:
    """ - for plotly figures only. """
    
    if by_mmHg:
        fig.update_xaxes( #tickprefix="At ",   # Dr. WJ and Ashkhan didn't like it
                         ticksuffix="mmHg", showtickprefix="all", showticksuffix="all", tickfont=dict(size=16),
                        mirror=True, linewidth=2, 
                        title=dict(text="<b>Applied Circumferential Pressure</b>", font=dict(size=20, family="Arial Black")),
                        )
        fig.update_yaxes(tickformat=".0%", tickwidth=2,  nticks=21, ticklabelstep=4,
                        mirror="ticks", linewidth=2, range=(0,1), 
                        title=dict(text="<b>Obstruction of<br>Field of View (S.E.)</b>",font=dict(size=18, family="Arial Black")), 
                        #title=dict(text="Width Obstructed of<br>Field of View (S.E.)",font=dict(size=18, family="Arial Black")), 
                        showgrid=True, gridcolor="#DDD", 
                        showspikes=True, spikemode="across", spikethickness=2, spikedash="solid", # ticklabelposition="inside top",
                        )
    #fig.update_traces(textangle=0, textposition="outside", cliponaxis=False)
    fig.update_layout(
        font=dict(
            family="Arial",
            size=16,
            color="black",
        ),
        title={
            "y":1,
            "x":0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font":dict(size=16)
        }, 
        width=width, height=height,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            title={"font_family": "Arial Black",},
            yanchor="middle",
            y=0.5,
            xanchor="center",
            x=0.08,
            #bgcolor="LightSteelBlue",
            bordercolor="Black", #font_size=16,
            borderwidth=2,
        ), 
        bargap=0.05, bargroupgap=0.0,
        dragmode="drawopenpath",
        newshape_line_color="cyan",
    )

    if textfont_color is None:
        if isinstance(textposition, (list, tuple, set, np.ndarray, pd.Series) ):
            textfont_color = ["#FFF" if textposition_each == "inside" else "#000" for textposition_each in textposition]
            print(textfont_color)
        elif textposition == "inside":
            textfont_color="#FFF"
        else:
            textfont_color="#000"
    fig.update_traces(textfont_size=16, textangle=0, textfont_color=textfont_color, 
                      textposition=textposition, cliponaxis=False, #textfont_family="Courier",
                      marker_line_color="#000", marker_line_width=2
                    )
    if by_mmHg:
        if textposition == "inside":
            fig.update_traces(texttemplate=[None]+[("&nbsp;"*space_ct)+("<br>"*br_ct)+"<b>%{y:.1%}</b>"]*5,)
        else:
            fig.update_traces(texttemplate=[None]+["<b>%{y:.1%}</b>"+("<br>"*br_ct)+("&nbsp;"*space_ct)]*5,)
            

    config = {
        "toImageButtonOptions" : {
            "format": "png", # one of png, svg, jpeg, webp
            "filename": 'custom_image',
            "scale": default_plotly_save_scale # Multiply title/legend/axis/canvas sizes by this factor
        },
        "modeBarButtonsToAdd": ["drawline","drawopenpath","drawclosedpath","drawcircle","drawrect","eraseshape"]
    }

    return config





# # Plotting

# In[ ]:


df_agg_long


# ## Plot varying Speculum Type

# In[ ]:


df_agg_long_flat.columns = [ col.replace("wd.","width.") for col in df_agg_long_flat.columns]
df_agg_long_flat.columns = [ col.replace("ht.","wd.") for col in df_agg_long_flat.columns]
df_agg_long_flat.columns = [ col.replace("width.","ht.") for col in df_agg_long_flat.columns]

df_agg_long_flat.columns = [ col.replace("wd_rel.","width_rel.") for col in df_agg_long_flat.columns]
df_agg_long_flat.columns = [ col.replace("ht_rel.","wd_rel.") for col in df_agg_long_flat.columns]
df_agg_long_flat.columns = [ col.replace("width_rel.","ht_rel.") for col in df_agg_long_flat.columns]


# In[ ]:





# In[ ]:


df_agg_long_flat.columns


# In[ ]:


#criteria = {"Material":["Nitrile","None"], "Method":["Middle","None"]}
criteria = {"Day Ct": 2}
varying = "Speculum Type"

df_sampled = filter_by_criteria(criteria,df_agg_long_flat)


fig = px.bar(df_sampled, 
             x="mmHg",y="wd_rel.mean", error_y="wd_rel.sem",
             color=varying, pattern_shape=varying, 
             color_discrete_map=color_discrete_map, pattern_shape_map=pattern_shape_map,
             barmode="group", #text=[".1%<br><br> " for a in range(18)],
             hover_data=["Speculum Type","Day Ct"],
             title=f"Varying {varying} with " + criteria_to_str(criteria), 
             category_orders=category_orders, labels=labels, template="simple_white", 
             )
config = customize_figure(fig, width=1100, height=300)

fig.for_each_trace( lambda trace: trace.update(marker=dict(color="#000",opacity=0.33,pattern=dict(shape=""))) if trace.name == "None" else (), )

fig.show(config=config)
fig.update_layout(title=dict(text=""))
save_plotly_figure(fig, file_name=f"Fig 4- Across {varying}- " + criteria_to_str(criteria) )


# In[ ]:


criteria = {"Material":["Nitrile"], "Method":["Middle","None"], "Size":["M"], "Speculum Type":["White"]}
varying = "Day Ct"

df_sampled = filter_by_criteria(criteria,df_agg_long_flat)
df_sampled["Day Ct"] = df_sampled["Day Ct"].astype(str)


fig = px.bar(df_sampled, 
             x="mmHg",y="wd_rel.mean", error_y="wd_rel.sem",
             color=varying, pattern_shape=varying, 
             color_discrete_map=color_discrete_map, pattern_shape_map=pattern_shape_map,
             barmode="group", #text=[".1%<br><br> " for a in range(18)],
             hover_data=["Speculum Type","Day Ct"],
             title=f"Varying {varying} with " + criteria_to_str(criteria), 
             category_orders=category_orders, labels=labels, template="simple_white", 
             )
config = customize_figure(fig, width=1100, height=300)

fig.for_each_trace( lambda trace: trace.update(marker=dict(color="#000",opacity=0.33,pattern=dict(shape=""))) if trace.name == "None" else (), )

fig.show(config=config)
fig.update_layout(title=dict(text=""))
save_plotly_figure(fig, file_name=f"Fig 4- Across {varying}- " + criteria_to_str(criteria) )


# In[ ]:


df_sampled = df_long.loc[ (df_long["Material"]=="Nitrile") & (df_long["Method"]=="Middle") & (df_long["Size"]=="M")  & (df_long["Day Ct"]==2) ]
df_sampled["Day Ct"] = df_sampled["Day Ct"].astype(str)
df_sampled["Set Trial Ct"] = df_sampled["Set Trial Ct"].astype(str)

fig = px.bar(df_sampled, 
             x="mmHg", y="wd_rel",  
             text_auto=".1%", barmode='group', color="Set Trial Ct",
             title="Speculum View Width - Specific Trials", 
             hover_data=["Size","Material","Method","Set Trial Ct"],
             category_orders=category_orders,
             labels=labels,
             color_discrete_map={"1": "Lightgray", "2": "Darkgray", "3": "Gray"},
             template="simple_white"
)


fig.update_layout(width=500, height=300)

fig.show()
save_plotly_figure(fig, file_name="Basic, all trials", scale=4)


# ## Plot vertical heights

# In[ ]:


colors={
    "None": "black",
    "<i>Durex</i><br>Condom": "blue",
    "<i>Lifestyle</i><br>Condom": "blue",
    "<i>Skyn</i><br>Condom": "blue",
    "<i>Trojan</i><br>Condom": "blue",
    "Medium<br><i>Vinyl</i><br>Glove": "orange",
    "<i>Large</i><br>Nitrile<br>Glove": "orange",
    "Medium<br>Nitrile<br>Glove": "orange",
    "Medium<br>Nitrile<br>Glove,<br><i>2 fingers</i>": "orange",
    "<i>Small</i><br>Nitrile<br>Glove": "orange",
    "<i>Small</i><br>Nitrile<br>Glove,<br><i>Palm</i>": "orange",
    "<i>Medium</i><br>Nitrile<br>Glove,<br><i>Palm</i>": "orange",
}


# In[ ]:


criteria = {"mmHg":[0,1], "Spec Ang":[3,5], "Day Ct":[2]}
varying = "Material"

df_sampled = df_agg_long_flat.loc[ np.all([ (type(val)!=list and df_agg_long[arg]==val ) or np.in1d(df_agg_long[arg],val)  for arg, val in criteria.items()], axis=0) ]
df_sampled = df_sampled.sort_values(["Opening Height.mean"]).reset_index()
df_sampled["Spec Ang"] = df_sampled["Spec Ang"].astype(str)  # makes discrete color plotting and string concatenation easier
df_sampled["name"] = df_sampled["Size"] + "-" + df_sampled["Material"] + "-"  + df_sampled["Material Type"] + "-"  + df_sampled["Method"] + "-"  + df_sampled["Spec Ang"]

extra_trials = speculum_df_raw.loc[speculum_df_raw["Filename"]=="None"].copy()
extra_trials = extra_trials.drop(extra_trials[extra_trials["Spec Ang"] == 4].index)
extra_trials["Opening Height.mean"] = extra_trials["Opening Height"]
extra_trials["Opening Height.sem"] = None
with_extra = pd.concat([df_sampled,extra_trials])
with_extra = with_extra.drop(columns=[col for col in with_extra if col not in df_sampled.columns])

df_sampled = with_extra
df_sampled["Spec Ang"] = df_sampled["Spec Ang"].astype(str)  # makes discrete color plotting and string concatenation easier
df_sampled["name"] = df_sampled["Speculum Type"]


names={
    "S-Nitrile-Glove-Palm":                 "<i>Small</i><br>Nitrile<br>Glove,<br><i>Palm</i>",
    "M-Nitrile-Glove-Palm":                 "<i>Medium</i><br>Nitrile<br>Glove,<br><i>Palm</i>",
    "None-None-None-None":                  "None", #"None<br>(3 clicks)",
    "Unspecified-Durex-Condom-Precut":      "<i>Durex</i><br>Condom",
    "Unspecified-Lifestyle-Condom-Precut":  "<i>Lifestyle</i><br>Condom",
    "Unspecified-Skyn-Condom-Precut":       "<i>Skyn</i><br>Condom",
    "Unspecified-Trojan-Condom-Precut":     "<i>Trojan</i><br>Condom",
    "L-Nitrile-Glove-Middle":               "<i>Large</i><br>Nitrile<br>Glove",
    "M-Nitrile-Glove-Middle":               "Medium<br>Nitrile<br>Glove",
    "S-Nitrile-Glove-Middle":               "<i>Small</i><br>Nitrile<br>Glove",
    "M-Nitrile-Glove-Two":                  "Medium<br>Nitrile<br>Glove,<br><i>2 fingers</i>",
    "M-Vinyl-Glove-Middle":                 "Medium<br><i>Vinyl</i><br>Glove",
}
names = {key: value.replace("<i>","").replace("</i>","") for key, value in names.items()}
colors={
    "None": "black",
    "<i>Durex</i><br>Condom": "blue",
    "<i>Lifestyle</i><br>Condom": "blue",
    "<i>Skyn</i><br>Condom": "blue",
    "<i>Trojan</i><br>Condom": "blue",
    "Medium<br><i>Vinyl</i><br>Glove": "orange",
    "<i>Large</i><br>Nitrile<br>Glove": "orange",
    "Medium<br>Nitrile<br>Glove": "orange",
    "Medium<br>Nitrile<br>Glove,<br><i>2 fingers</i>": "orange",
    "<i>Small</i><br>Nitrile<br>Glove": "orange",
    "<i>Small</i><br>Nitrile<br>Glove,<br><i>Palm</i>": "orange",
    "<i>Medium</i><br>Nitrile<br>Glove,<br><i>Palm</i>": "orange",
}
df_sampled["name_formatted"] = df_sampled["name"].replace(names, value=None)  # values=None indicates that names has the values in it (i.e is a dict, not a list)
df_sampled["colors"] = df_sampled["name_formatted"].replace(colors, value=None)
#df_sampled["name"] = df_sampled["name_formatted"].replace(names)

fig = px.bar(df_sampled, 
             #x = np.argsort(df_sampled["Opening Height.mean"]),
             x = "name_formatted",
             y="Opening Height.mean", error_y="Opening Height.sem", 
             category_orders={**category_orders, }, 
             labels={**labels,"Spec Ang":"<b>Number of Clicks Open</b>", "Opening Height.mean":"<b>Initial Height of <br>Speculum Opening</b>"}, 
             template="simple_white", 
             hover_data=["Size","Material","Method","name"], #color = ["gray","gray","red","gray","gray"]
             color="Material Type", color_discrete_map={
                 "Glove": px.colors.qualitative.D3[0], #"navy",
                 "Condom": px.colors.qualitative.D3[1], #"maroon",
                 "None":"darkgray"},
             pattern_shape="Material Type", pattern_shape_map={
                 "Glove": "/",
                 "Condom": "\\",
                 "None":""},
             )
fig.update_xaxes(matches=None)
fig.update_traces(texttemplate=""" <br><b>%{y:.2f}<br>cm</b>""", textposition="outside",)

fig.update_xaxes(linewidth=2, #showticklabels=False, nticks=0,
                 title=dict(
                     #text="Speculum-Material Combination",
                     text="<b>Speculum Sheath</b>",
                     #text="",
                     font=dict(size=18, family="Arial Black")
                     ),
                 )
fig.update_yaxes(ticksuffix="cm", tickformat=".0f", tickwidth=2, range=(0,6),  nticks=6*2+1, ticklabelstep=2,
                mirror=True, linewidth=2,
                #title=dict(text="Initial Height of <br>Speculum Opening", font=dict(size=18, family="Arial Black")), 
                title=dict(font=dict(size=18, family="Arial Black")), 
                showgrid=True, gridcolor="#AAA", 
                showspikes=True, spikemode="across", spikethickness=2, spikedash="solid", # ticklabelposition="inside top",
                )



config = customize_figure(fig, width=1200, height=500, by_mmHg=False)

#fig.update_layout(showlegend=False)
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="middle",
    y=0.85,
    xanchor="center",
    x=0.25,
    )
)

fig.show(config=config)

save_plotly_figure(fig, file_name=f"Fig 3- Vertical Height Bar Plot" )


# In[ ]:




