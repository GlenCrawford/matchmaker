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
# education
# ethnicity
# height
# job
# location
# offspring
# pets
# religion
# smokes
# speaks

INPUT_DATA_COLUMNS_TO_USE = [
  'age', # Numeric, min: 18, max: 110, mean: 32.3, std: 9.4, no missing values.
  'relationship_status', # String, no missing values, 5 unique values: single (93%), seeing someone (3%), available (3%), married (0.5%), unknown (0.02%). Don't one-hot encode this, pass this through exactly as we're likely going to do a direct value look-up.
  'sex', # String, no missing values, 2 unique values: m (59.8%), f (40.2%). Don't one-hot encode this, pass this through exactly as we're likely going to do a direct value look-up.
  'sexual_orientation', # String, no missing values, 3 unique values: straight (86%), gay (9.3%), bisexual (4.6%). Don't one-hot encode this, pass this through exactly as we're likely going to do a direct value look-up.
  'body_type', # String, 5,293 missing values, 12 unique values: average, athletic, etc. Consolidate similar values to reduce unique values (and replace missing ones) to fit, average, curvy, thin, overweight, and unknown. And one-hot encode.
  'diet', # String, 24,389 missing values (41%), 18 unique values: anything, kosher, etc. Consolidate similar values to reduce unique values (and replace missing ones) to anything, vegetarian, kosher, etc. And one-hot encode.
  'drinks', # String, 2,985 missing values (5%), 6 unique values: often, rarely, etc. Consolidate similar values to reduce unique values (and replace missing ones). And one-hot encode.
  'drugs' # String, 14,076 missing values (23%), 3 unique values: never, sometimes, often. Replace missing values with never, which is not a great assumption, but whatever. And one-hot encode.
]

# Input data columns not using:
# * essay0, essay1, essay2, essay3, essay4, essay5, essay6, essay7, essay8, essay9: Skip these for now, but intend to use later, by layering some NLP similarity on top of the more basic features.
# * last_online, sign: Not useful.
# * income: Would be interesting, but it's apparently an optional field: 81% of all rows have a value of -1. Not enough to be useful.
# * ?: ?

NUMERIC_COLUMNS_TO_SCALE = ['age']
CATEGORICAL_COLUMNS_TO_ONE_HOT_ENCODE = ['body_type', 'diet', 'drinks', 'drugs']

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
      },
      'diet': {
        'mostly anything': 'anything',
        'strictly anything': 'anything',
        'mostly vegetarian': 'vegetarian',
        'strictly vegetarian': 'vegetarian',
        'mostly vegan': 'vegan',
        'strictly vegan': 'vegan',
        'mostly kosher': 'kosher',
        'strictly kosher': 'kosher',
        'mostly halal': 'halal',
        'strictly halal': 'halal',
        'mostly other': 'other',
        'strictly other': 'other',
        np.nan: 'anything'
      },
      'drinks': {
        'very often': 'often',
        'not at all': 'never',
        'desperately': 'often', # I presume this means often.
        np.nan: 'unknown'
      },
      'drugs': {
        np.nan: 'never'
      }
    }
  )

  # Apply one-hot encoding to categorical features.
  data_frame = pd.get_dummies(
    data_frame,
    columns = CATEGORICAL_COLUMNS_TO_ONE_HOT_ENCODE,
    sparse = False
  )

  # Scale/normalize numeric columns to between 0 and 1.
  numeric_scaler = sklearn.preprocessing.MinMaxScaler(feature_range = (0, 1), copy = True)
  data_frame[NUMERIC_COLUMNS_TO_SCALE] = numeric_scaler.fit_transform(data_frame[NUMERIC_COLUMNS_TO_SCALE].to_numpy())

  return data_frame
