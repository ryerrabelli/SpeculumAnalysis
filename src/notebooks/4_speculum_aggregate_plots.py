#!/usr/bin/env python
# coding: utf-8

# # Setup

# ## Pip install

# In[1]:


# Don't forget to restart runtime after installing

get_ipython().run_line_magic('pip', 'install "labelbox[data]" --quiet')
get_ipython().run_line_magic('pip', 'install -U kaleido  --quiet # for saving the still figures')
get_ipython().run_line_magic('pip', 'install poppler-utils   # for exporting to .eps extension')
get_ipython().run_line_magic('pip', 'install plotly==5.7.0.    # need 5.7.0, not 5.5, so I can use ticklabelstep argument')
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


# In[3]:



notebook_filename = requests.get("http://172.28.0.2:9000/api/sessions").json()[0]["name"]

# Avoids scroll-in-the-scroll in the entire Notebook
def resize_colab_cell():
  display(IPython.display.Javascript('google.colab.output.setIframeHeight(0, true, {maxHeight: 10000})'))
get_ipython().events.register('pre_run_cell', resize_colab_cell)


#@markdown ### func `def get_path_to_save(...):`
def get_path_to_save(plot_props:dict=None, file_prefix="", save_filename:str=None, save_in_subfolder:str=None, extension="jpg", create_folder_if_necessary=True):
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
    return os.path.join(save_path, file_prefix+save_filename+"."+extension)
    #plt.savefig(os.path.join(save_path, save_filename+"."+extension))


# In[4]:


#@title ## Mount google drive and import my code

mountpoint_folder_name = "drive"  # can be anything, doesn't have to be "drive"
project_path_within_drive = "PythonProjects/SpeculumAnalysis" #@param {type:"string"}
#project_path_within_drive = "UIUC ECs/Rahul_Ashkhan_Projects/SpeculumProjects_Shared/Analysis" #@param {type:"string"}
project_path_full = os.path.join("/content/",mountpoint_folder_name,
                        "MyDrive",project_path_within_drive)

get_ipython().run_line_magic('cd', '{project_path_full}')


# In[5]:


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


# # Skip ahead from loaded code

# In[6]:


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

# ## Setup dicts and helper functions

# In[250]:


category_orders={
    #"Size": ["S", "M", "L","Unspecified","None"],
    #"Size": ["S","Small", "M", "Medium", "L", "Large","Unspecified","None"],
    "Size": ["Small", "Medium", "Large","Unspecified","None", "S", "M", "L"],  # change the order of S vs Small etc changes the color
    "Material":["Nitrile","Vinyl","Trojan", "Lifestyle", "Durex", "Skyn","None"],
    "Material Type":["Glove","Condom","None"],
    "Method":["Middle","Two","Palm","Middle finger","Two fingers","Palm","Precut","None"],
    "Speculum Type":["White","Green"]
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

# In[8]:


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


# # Plotting

# ## Plot aggregates across trials

# #### Setup for plotting aggregates

# In[134]:


hasattr("abc","__iter__")


# In[238]:


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





# ### Gloves

# #### Glove sizes

# In[251]:


#criteria = {"Material":["Nitrile","None"], "Method":["Middle","None"]}
criteria = {"Material":"Nitrile", "Method":"Middle"}
varying = "Size"

df_sampled = filter_by_criteria(criteria,df_agg_long_flat)
df_sampled["Size"] = df_sampled["Size"].replace({"S":"Small", "M":"Medium", "L":"Large"})


fig = px.bar(df_sampled, 
             x="mmHg",y="wd_rel.mean", error_y="wd_rel.sem", #error_y_minus=[0]*18, 
             color=varying, pattern_shape=varying, 
             #color_discrete_sequence=px.colors.qualitative.Safe, pattern_shape_sequence=["/", "+", "\\"], 
             color_discrete_map=color_discrete_map, pattern_shape_map=pattern_shape_map,
             barmode="group", #text=[".1%<br><br> " for a in range(18)],
             hover_data=["Size","Material","Method"],
             title=f"Varying {varying} with " + criteria_to_str(criteria), 
             category_orders=category_orders, labels=labels, template="simple_white", 
             )
#fig.update_traces(hovertemplate="""%{x}""") #
config = customize_figure(fig, width=1100, height=300)

fig.for_each_trace( lambda trace: trace.update(marker=dict(color="#000",opacity=0.33,pattern=dict(shape=""))) if trace.name == "None" else (), )

fig.show(config=config)
fig.update_layout(title=dict(text=""))
save_plotly_figure(fig, file_name=f"Fig 4- Across {varying}- " + criteria_to_str(criteria) )


# #### Glove material

# In[258]:


#criteria = {"Size":["M","None"], "Method":["Middle","None"]}
criteria = {"Size":"M", "Method":"Middle"}

varying = "Material"

df_sampled = filter_by_criteria(criteria,df_agg_long_flat)
fig = px.bar(df_sampled, 
             x="mmHg",y="wd_rel.mean", error_y="wd_rel.sem", 
             color=varying, pattern_shape=varying, 
             #color_discrete_sequence=px.colors.qualitative.Safe[6:-1:], pattern_shape_sequence=["x", "+", "\\"], 
             color_discrete_map=color_discrete_map, pattern_shape_map=pattern_shape_map,
             barmode="group", #text=[".1%<br><br> " for a in range(18)],
             hover_data=["Size","Material","Method"],
             title=f"Varying {varying} with " + criteria_to_str(criteria), 
             category_orders=category_orders, labels={**labels,"Material":"<b>Glove<br>Material</b>"}, template="simple_white", 
             )

config = customize_figure(fig, width=1100, height=300, textposition="outside", space_ct=3)

#fig.update_traces(texttemplate=["<br>"+"<b>%{y:.1%}</b>", "<b>%{y:.1%}</b>"+("&nbsp;"*10)+("<br>"*1)]*5)

fig.for_each_trace( lambda trace: trace.update(
    textfont_color="#000",
    textposition="outside",
    cliponaxis=False,
    texttemplate=[None]+["&nbsp;"*10+"<b>%{y:.1%}</b><br> "]*5,
    ) if trace.name == "Vinyl" else (), )

fig.for_each_trace( lambda trace: trace.update(marker=dict(color="#000",opacity=0.33,pattern=dict(shape=""))) if trace.name == "None" else (), )


fig.show(config=config)
fig.update_layout(title=dict(text=""))
save_plotly_figure(fig, file_name=f"Fig 5- Across {varying}- " + criteria_to_str(criteria) )


# #### Glove method

# In[247]:


#criteria = {"Size":["M","None"], "Material":["Nitrile","None"]}
criteria = {"Size":"M", "Material":"Nitrile"}

varying = "Method"

df_sampled = filter_by_criteria(criteria,df_agg_long_flat)
df_sampled["Method"] = df_sampled["Method"].replace({"Middle":"Middle finger","Two":"Two fingers"})
fig = px.bar(df_sampled, 
             x="mmHg",y="wd_rel.mean", error_y="wd_rel.sem", 
             color=varying, pattern_shape=varying, 
             #color_discrete_sequence=px.colors.qualitative.D3, pattern_shape_sequence=["x", "+", "-"], 
             color_discrete_map=color_discrete_map, pattern_shape_map=pattern_shape_map,             
             barmode="group", #text=[".1%<br><br> " for a in range(18)],
             hover_data=["Size","Material","Method","wd_rel.amin","wd_rel.median","wd_rel.amax"],
             #title=f"Varying {varying} with " + criteria_to_str(criteria), 
             category_orders=category_orders, labels=labels, template="simple_white", 
             )

config = customize_figure(fig, width=1100, height=300)

fig.for_each_trace( lambda trace: trace.update(marker=dict(color="#000",opacity=0.33,pattern=dict(shape=""))) if trace.name == "None" else (), )


fig.show(config=config)
fig.update_layout(title=dict(text=""))
save_plotly_figure(fig, file_name=f"Fig 6- Across {varying}- " + criteria_to_str(criteria) )


# Relative Obstruction of Field of View
# 
# Percent Field of View Obstructed

# ### Plot condoms

# In[254]:


#criteria = {"Material Type":"Condom"}
criteria = {"Material Type":["Condom","None"]}
varying = "Material"

#colors = px.colors.qualitative.Plotly.copy() #[0:4]+["black"]
#colors = ['black']*10
#colors[8] = "black"
#colors = ['#636EFA', '#EF553B',"Black", '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
#colors={"Trojan":"red","gray":"blue","None":"green","Lifestyle":"green","Skyn":"blue"}
#df_sampled = df_agg_long_flat.loc[ np.all([df_agg_long[arg]==val for arg, val in criteria.items()], axis=0) ]
df_sampled = filter_by_criteria(criteria,df_agg_long_flat)
#df_sampled["color"] = df_sampled["Material"].copy().replace(colors)
colors = px.colors.qualitative.Safe #[0:4]+["black"]
fig = px.bar(df_sampled, 
             x="mmHg",y="wd_rel.mean", error_y="wd_rel.sem", 
             color=varying, pattern_shape=varying, 
             #color_discrete_sequence=px.colors.qualitative.Safe, pattern_shape_sequence=["/","|", "-", "\\"], 
             color_discrete_map=color_discrete_map, pattern_shape_map=pattern_shape_map,
             barmode="group", #text=[".1%<br><br> " for a in range(18)],
             hover_data=["Size","Material","Method"],
             title=f"Varying {varying} with " + criteria_to_str(criteria), 
             category_orders=category_orders, labels={**labels,"Material":"<b>Condom<br>Brand</b>"}, template="simple_white", 
             )

config = customize_figure(fig, width=1100, height=300)

for idx, trace in enumerate(fig["data"]):
    trace["name"] = trace["name"].split()[-1]


#fig.for_each_trace( lambda trace: trace.update(marker=dict(color="#000",opacity=0.33,pattern=dict(shape=""))) if trace.name == "None" else (), )
fig.for_each_trace( lambda trace: trace.update(marker=dict(color="#AAA",pattern=dict(shape="")),textfont_color="#FFF") if trace.name == "None" else (), )
#fig.for_each_trace( lambda trace: trace.update(marker=dict(color="#000",pattern=dict(shape="x",fillmode="replace"))) if trace.name == "None" else (), )

#fig.update_traces(texttemplate=[None]+[""" <br><br><br><b>%{y:.1%}</b>"""]*5,)
fig.update_traces(texttemplate=[None]+["<b>%{y:.1%}</b>"+("&nbsp;"*7)]*5,)

fig.update_traces(textangle=-90,)

fig.show(config=config)
fig.update_layout(title=dict(text=""))
save_plotly_figure(fig, file_name=f"Fig 7- Across {varying}- " + criteria_to_str(criteria) )


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


#[key for key in df_sampled["name_formatted"] if key not in colors.keys()]
len(df_sampled["name_formatted"])


# In[ ]:


list(names.values())[0:-1:]


# In[249]:


criteria = {"mmHg":[0,1], "Spec Ang":[3,5]}
varying = "Material"

df_sampled = df_agg_long_flat.loc[ np.all([ (type(val)!=list and df_agg_long[arg]==val ) or np.in1d(df_agg_long[arg],val)  for arg, val in criteria.items()], axis=0) ]
df_sampled = df_sampled.sort_values(["Vertical Height.mean"]).reset_index()
df_sampled["Spec Ang"] = df_sampled["Spec Ang"].astype(str)  # makes discrete color plotting and string concatenation easier
df_sampled["name"] = df_sampled["Size"] + "-" + df_sampled["Material"] + "-"  + df_sampled["Material Type"] + "-"  + df_sampled["Method"] + "-"  + df_sampled["Spec Ang"]

extra_trials = speculum_df_raw.loc[speculum_df_raw["Filename"]=="None"].copy()
extra_trials = extra_trials.drop(extra_trials[extra_trials["Spec Ang"] == 4].index)
extra_trials["Vertical Height.mean"] = extra_trials["Vertical Height"]
extra_trials["Vertical Height.sem"] = None
with_extra = pd.concat([df_sampled,extra_trials])
with_extra = with_extra.drop(columns=[col for col in with_extra if col not in df_sampled.columns])

df_sampled = with_extra
df_sampled["Spec Ang"] = df_sampled["Spec Ang"].astype(str)  # makes discrete color plotting and string concatenation easier
df_sampled["name"] = df_sampled["Size"] + "-" + df_sampled["Material"] + "-"  + df_sampled["Material Type"] + "-"  + df_sampled["Method"] #+ "-"  + df_sampled["Spec Ang"]

names={
    "None-None-None-None-3": "None", #"None<br>(3 clicks)",
    "None-None-None-None-5": "None", #"None<br>(5 clicks)",
    "Unspecified-Durex-Condom-Precut-3": "<i>Durex</i><br>Condom",
    "Unspecified-Lifestyle-Condom-Precut-3": "<i>Lifestyle</i><br>Condom",
    "Unspecified-Skyn-Condom-Precut-3": "<i>Skyn</i><br>Condom",
    "Unspecified-Trojan-Condom-Precut-3": "<i>Trojan</i><br>Condom",
    "M-Vinyl-Glove-Middle-3": "Medium<br><i>Vinyl</i><br>Glove",
    "L-Nitrile-Glove-Middle-5": "<i>Large</i><br>Nitrile<br>Glove",
    "M-Nitrile-Glove-Middle-5": "Medium<br>Nitrile<br>Glove",
    "M-Nitrile-Glove-Two-5": "Medium<br>Nitrile<br>Glove,<br><i>Two-fingers</i>",
    "S-Nitrile-Glove-Middle-5": "<i>Small</i><br>Nitrile<br>Glove"
}
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
             #x = np.argsort(df_sampled["Vertical Height.mean"]),
             x = "name_formatted",
             y="Vertical Height.mean", error_y="Vertical Height.sem", 
             category_orders={**category_orders, }, 
             labels={**labels,"Spec Ang":"<b>Number of Clicks Open</b>", "Vertical Height.mean":"<b>Initial Height of <br>Speculum Opening</b>"}, 
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
             facet_col="Spec Ang", # facet_row="Material Type",
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

fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')


fig.add_annotation(x=4, y=0.25,
            text="<b>Failed: <i>Broke speculum</i></b>",
            textangle=-90, xanchor="center", yanchor="bottom",
            font=dict(color="#3976af"), bgcolor="white",  # 3976af is blue
            showarrow=False,
            row=1, col=2)
fig.add_annotation(x=5, y=0.25,
            text="<b><i>Failed: <i>Slipped off speculum</i></b>",
            textangle=-90, xanchor="center", yanchor="bottom",
            font=dict(color="#3976af"), bgcolor="white",
            showarrow=False,
            row=1, col=2)
fig.add_annotation(x=6, y=0.25,
            text="<b><i>Failed: <i>Slipped off speculum</i></b>",
            textangle=-90, xanchor="center", yanchor="bottom",
            font=dict(color="#3976af"), bgcolor="white",
            showarrow=False,
            row=1, col=2)

#fig.for_each_trace( lambda trace: trace.update(marker=dict(color="orange")) if "Glove" in trace.name else (), )
#fig.for_each_trace( lambda trace: trace.update(marker=dict(color="#000",opacity=0.33,pattern=dict(shape="")), color="black") if trace.name == "None" else (), )


for index in range(2):
    #fig.layout.annotations[index]["text"] = f"<b>With {[3,5][index]} (of 7) Clicks Open</b>"
    fig.layout.annotations[index]["text"] = f"<b>With {[3,5][index]} Clicks of Angular Adjustment"

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


#df_sampled[["name","name_formatted"]]
df_sampled


# In[ ]:




