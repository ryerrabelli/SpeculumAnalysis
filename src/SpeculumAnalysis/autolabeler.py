import json
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import labelbox

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