import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing

# Relative from the root directory.
INPUT_DATA_PATH = 'data/okcupid_profiles.csv'

INPUT_DATA_COLUMN_NAMES = [
  'age', 'relationship_status', 'sex', 'sexual_orientation', 'body_type', 'diet', 'drinks', 'drugs', 'education',
  'ethnicity', 'height', 'income', 'job', 'last_online', 'location', 'offspring', 'pets', 'religion', 'sign', 'smokes',
  'speaks', 'essay0', 'essay1', 'essay2', 'essay3', 'essay4', 'essay5', 'essay6', 'essay7', 'essay8', 'essay9'
]

# TODO:
# job
# location
# offspring
# pets
# religion

INPUT_DATA_COLUMNS_TO_USE = [
  'age', # Numeric, min: 18, max: 110, mean: 32.3, std: 9.4, no missing values.
  'relationship_status', # String, no missing values, 5 unique values: single (93%), seeing someone (3%), available (3%), married (0.5%), unknown (0.02%). Don't one-hot encode this, pass this through exactly as we're likely going to do a direct value look-up.
  'sex', # String, no missing values, 2 unique values: m (59.8%), f (40.2%). Don't one-hot encode this, pass this through exactly as we're likely going to do a direct value look-up.
  'sexual_orientation', # String, no missing values, 3 unique values: straight (86%), gay (9.3%), bisexual (4.6%). Don't one-hot encode this, pass this through exactly as we're likely going to do a direct value look-up.
  'body_type', # String, 5,293 missing values, 12 unique values: average, athletic, etc. Consolidate similar values to reduce unique values (and replace missing ones) to fit, average, curvy, thin, overweight, and unknown. And one-hot encode.
  'diet', # String, 24,389 missing values (41%), 18 unique values: anything, kosher, etc. Consolidate similar values to reduce unique values (and replace missing ones) to anything, vegetarian, kosher, etc. And one-hot encode.
  'drinks', # String, 2,985 missing values (5%), 6 unique values: often, rarely, etc. Consolidate similar values to reduce unique values (and replace missing ones). And one-hot encode.
  'drugs', # String, 14,076 missing values (23%), 3 unique values: never, sometimes, often. Replace missing values with never, which is not a great assumption, but whatever. And one-hot encode.
  'education', # String, 6,625 missing values (11%), 32 unique values: high school, college/university, etc. Consolidate similar values to reduce unique values (and replace missing ones). And one-hot encode.
  'ethnicity', # String, 5,676 missing values (9.5%), 217 unique values: white, asian, etc. Consolidate similar values to reduce unique values (and replace missing ones). 81% of all non-missing values are white, asian, hispanic/latin or black. Lump missing and everything else into unknown. Controversial... And one-hot encode.
  'smokes', # String. 5,511 missing values (9.2%), 5 unique values: no, sometimes, etc. Map the values to binary 0 and 1, where any degree of smoking is 1.
  'speaks' # String, 50 missing values (~0%), 7,643 missing values: english, etc. Consolidate similar values to most common ones to reduce unique values (and replace missing ones), and trim to one each, prioritizing the first mentioned one. And one-hot encode.
]

# Input data columns not using:
# * essay0, essay1, essay2, essay3, essay4, essay5, essay6, essay7, essay8, essay9: Skip these for now, but intend to use later, by layering some NLP similarity on top of the more basic features.
# * last_online, sign: Not useful.
# * income: Would be interesting, but it's apparently an optional field: 81% of all rows have a value of -1. Not enough to be useful.
# * height: Not really relevant for matchmaking. Is also going to correlate with sex and create noise in that sense. *Could* be untangled, as an academic exercise, but again, it's not relevant enough.

NUMERIC_COLUMNS_TO_SCALE = ['age']
CATEGORICAL_COLUMNS_TO_ONE_HOT_ENCODE = ['body_type', 'diet', 'drinks', 'drugs', 'education', 'ethnicity', 'speaks']

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
      },
      'education': {
        'dropped out of high school': 'less_than_high_school',
        'working on high school': 'less_than_high_school',
        'high school': 'high_school',
        'graduated from high school': 'high_school',
        'dropped out of two-year college': 'high_school',
        'dropped out of college/university': 'high_school',
        'dropped out of law school': 'high_school',
        'dropped out of med school': 'high_school',
        'two-year college': 'in_progress_study',
        'college/university': 'in_progress_study',
        'working on two-year college': 'in_progress_study',
        'working on college/university': 'in_progress_study',
        'law school': 'in_progress_study',
        'working on law school': 'in_progress_study',
        'working on med school': 'in_progress_study',
        'med school': 'in_progress_study',
        'graduated from two-year college': 'completed_undergraduate_study',
        'graduated from college/university': 'completed_undergraduate_study',
        'graduated from law school': 'completed_undergraduate_study',
        'dropped out of masters program': 'completed_undergraduate_study',
        'dropped out of ph.d program': 'completed_undergraduate_study',
        'masters program': 'completed_undergraduate_study',
        'working on masters program': 'completed_undergraduate_study',
        'working on ph.d program': 'completed_undergraduate_study',
        'ph.d program': 'completed_postgraduate_study',
        'graduated from masters program': 'completed_postgraduate_study',
        'graduated from ph.d program': 'completed_postgraduate_study',
        'graduated from med school': 'completed_postgraduate_study',
        'dropped out of space camp': 'unknown',
        'working on space camp': 'unknown',
        'space camp': 'unknown',
        'graduated from space camp': 'unknown',
        np.nan: 'unknown'
      },
      'ethnicity': {
        'hispanic / latin': 'hispanic_latin',
        np.nan: 'unknown',
      },
      'smokes': {
        'no': 0,
        'sometimes': 1,
        'when drinking': 1,
        'yes': 1,
        'trying to quit': 1,
        np.nan: 0 # Assume no, as 81% of all non-missing values are no, so it's by far the most likely value. Very imperfect rationale though.
      },
      'speaks': {
        np.nan: 'unknown'
      }
    }
  )

  data_frame = data_frame.replace(
    {
      'ethnicity': {
        r'^(?!white$|asian$|hispanic_latin$|black$).*': 'unknown' # I am *not* mapping 217 unique values...
      },
      # Believe it or not, this is enough to consolidate and trim down to each row having one normalized language (favouring their first listed) without any non-whitelisted values.
      'speaks': {
        r'^mandarin(.*)$': 'mandarin_chinese',
        r'^spanish(.*)$': 'spanish',
        r'^english(.*)$': 'english',
        r'^hindi(.*)$': 'hindi',
        r'^russian(.*)$': 'russian',
        r'^japanese(.*)$': 'japanese',
        r'^portuguese(.*)$': 'portuguese',
        r'^french(.*)$': 'french',
        r'^afrikaans(.*)$': 'afrikaans'
      }
    },
    regex = True
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
