import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing

# Relative from the root directory.
INPUT_DATA_PATH = 'data/okcupid_profiles.csv'

INPUT_DATA_COLUMN_NAMES = [
  'age', 'relationship_status', 'sex', 'sexual_orientation', 'body_type', 'diet', 'drinks', 'drugs', 'education', 'ethnicity', 'height', 'income', 'job', 'last_online', 'location', 'offspring', 'pets', 'religion', 'sign', 'smokes', 'speaks', 'essay0', 'essay1', 'essay2', 'essay3', 'essay4', 'essay5', 'essay6', 'essay7', 'essay8', 'essay9'
]

# TODO:
# diet
# drinks
# drugs
# education
# ethnicity
# height
# income
# job
# last_online
# location
# offspring
# pets
# religion
# sign
# smokes
# speaks
# essay0
# essay1
# essay2
# essay3
# essay4
# essay5
# essay6
# essay7
# essay8
# essay9

INPUT_DATA_COLUMNS_TO_USE = [
  'age', # Numeric, min: 18, max: 110, mean: 32.3, std: 9.4, no missing values.
  'relationship_status', # String, no missing values, 5 unique values: single (93%), seeing someone (3%), available (3%), married (0.5%), unknown (0.02%). Don't one-hot encode this, pass this through exactly as we're likely going to do a direct value look-up.
  'sex', # String, no missing values, 2 unique values: m (59.8%), f (40.2%). Don't one-hot encode this, pass this through exactly as we're likely going to do a direct value look-up.
  'sexual_orientation', # String, no missing values, 3 unique values: straight (86%), gay (9.3%), bisexual (4.6%). Don't one-hot encode this, pass this through exactly as we're likely going to do a direct value look-up.
  'body_type' # String, 5,293 missing values, 12 unique values: average, athletic, etc. Consolidate similar values to reduce uniquess values (and replace missing ones) to fit, average, curvy, thin, overweight, and unknown. And one-hot encode.
]

# NUMERIC_COLUMNS_TO_SCALE = []
CATEGORICAL_COLUMNS_TO_ONE_HOT_ENCODE = ['body_type']

def load_input_data():
  return pd.read_csv(
    INPUT_DATA_PATH,
    header = 0,
    names = INPUT_DATA_COLUMN_NAMES,
    usecols = INPUT_DATA_COLUMNS_TO_USE
  )

def preprocess_input_data(data_frame):
  # Drop rows where relationship_status = unknown.
  data_frame = data_frame[data_frame['relationship_status'] != 'unknown']

  # Consolidate similar body types from 12 unique values (plus missing) to 6.
  data_frame = data_frame.replace(
    {
      'body_type': {
        'athletic': 'fit',
        'skinny': 'thin',
        'jacked': 'fit',
        'full figured': 'curvy',
        'a little extra': 'curvy',
        'rather not say': 'unknown',
        'used up': 'unknown', # I don't even know what this means...
        np.nan: 'unknown'
      }
    }
  )

  # Apply one-hot encoding to categorical features.
  data_frame = pd.get_dummies(
    data_frame,
    columns = CATEGORICAL_COLUMNS_TO_ONE_HOT_ENCODE,
    sparse = False
  )

  return data_frame
