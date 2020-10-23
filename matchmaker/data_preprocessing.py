from . import utilities as Utilities

import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing

pd.set_option('display.min_rows', 25)

# Relative from the project root directory.
INPUT_DATA_PATH = 'data/okcupid_profiles.csv'

INPUT_DATA_COLUMN_NAMES = [
  'age', 'relationship_status', 'sex', 'sexual_orientation', 'body_type', 'diet', 'drinks', 'drugs', 'education',
  'ethnicity', 'height', 'income', 'job', 'last_online', 'location', 'offspring', 'pets', 'religion', 'sign', 'smokes',
  'speaks', 'essay0', 'essay1', 'essay2', 'essay3', 'essay4', 'essay5', 'essay6', 'essay7', 'essay8', 'essay9'
]

INPUT_DATA_COLUMNS_TO_USE = [
  # age:
  # Numeric
  # No missing values
  # Minimum: 18, maximum: 110, mean: 32.3, std: 9.4
  'age',

  # relationship_status:
  # String
  # No missing values
  # 5 unique values: single (93%), seeing someone (3%), available (3%), married (0.5%), unknown (0.02%)
  # Drop rows that are unknown, and it's kind of critical given the problem domain.
  # Drop rows that are seeing someone or married, because...why are they online dating?
  # Remove this feature after dropping the rows, won't need it after that.
  'relationship_status',

  # sex:
  # String
  # No missing values
  # 2 unique values: m (59.8%), f (40.2%)
  # Don't encode, pass this through exactly as we are going to use this for a direct value look-up.
  'sex',

  # sexual_orientation:
  # String
  # No missing values
  # 3 unique values: straight (86%), gay (9.3%), bisexual (4.6%)
  # Don't encode, pass this through exactly as we are going to use this for a direct value look-up.
  'sexual_orientation',

  # body_type:
  # String
  # 5,293 missing values (8.8%)
  # 12 unique values: average, athletic, etc.
  # Consolidate similar values to reduce unique values to thin, fit, average, curvy and overweight.
  # Replace missing and "rather not say" values with average.
  # Encode the final set of unique values to ordered and scaled ordinals.
  'body_type',

  # diet:
  # String
  # 24,389 missing values (41%)
  # 18 unique values: anything, kosher, etc.
  # Consolidate similar values to reduce unique values to vegan, vegetarian and anything.
  # There are a couple of religious diets here, namely kosher and halal. Don't use those here; we have a religion
  # feature that we'll use to factor religion in.
  # Replace missing values with anything. Can't really assume they're vegetarian unless they say so, etc.
  'diet',

  # drinks:
  # String
  # 2,985 missing values (5%)
  # 6 unique values: often, rarely, etc.
  # Consolidate similar values to reduce unique values (e.g. very often to often).
  # Replace missing values with socially as that is by far the most common value at 70%.
  # Encode the final set of unique values to ordered and scaled ordinals.
  'drinks',

  # drugs:
  # String
  # 14,076 missing values (23%)
  # 3 unique values: never, sometimes, often.
  # Replace missing values with never, which is not a great assumption, but whatever.
  # Encode the final set of unique values to ordered and scaled ordinals.
  'drugs',

  # education:
  # String
  # 6,625 missing values (11%)
  # 32 unique values: high school, college/university, etc.
  # Consolidate similar values to reduce unique values.
  # Replace missing and joke (space camp) values with high school. Assume that if someone has a degree they would have
  # said so.
  # Encode the final set of unique values to ordered and scaled ordinals.
  'education',

  # ethnicity:
  # String
  # 5,676 missing values (9.5%)
  # 217 unique values: white, asian, etc.
  # Consolidate similar values to reduce unique values. 81% of all non-missing values are white, asian, hispanic/latin
  # or black. Lump everything else into unknown. Controversial...
  # Replace missing values with unknown.
  # One-hot encode.
  'ethnicity',

  # offspring:
  # String
  # 35,554 missing values (59.3%)
  # 15 unique values: has kids, wants kids, etc.
  # Consolidate similar values to reduce unique values down to just no_kids, no_kids_dont_want_any, has_kids,
  # has_kids_but_no_more.
  # replace missing values with unknown.
  # One-hot encode.
  'offspring',

  # pets:
  # String
  # 19,916 missing values (33.2%)
  # 15 unique values: has dogs, likes cats, etc.
  # This is a strange one; it only includes cats and dogs. So if you have a rabbit but want a dog, you might be NaN, or
  # "likes dogs", and no mention of the rabbit.
  # Ignore the whole "likes" and "dislikes" aspect and simply turn it into a feature of what they actually have by first
  # mapping the values into a comma-separated list of what they have (e.g. "cats,dogs"), and then dropping and replacing
  # the column in favour of two separate columns (pets_cats and pets_dogs) which are then label-binarizer encoded.
  'pets',

  # religion:
  # String
  # 20,221 missing values (33.7%)
  # 45 unique values: christianity, judaism, etc.
  # Consolidate similar values to reduce unique values down to just atheism, agnosticism, buddhism, hinduism, islam,
  # judaism, christianity, catholicism, other.
  # Replace missing values with unknown.
  # One-hot encode.
  'religion',

  # smokes:
  # String
  # 5,511 missing values (9.2%)
  # 5 unique values: no, sometimes, etc.
  # Consolidate values down to a smaller number of unique values, e.g. "when drinking" to "sometimes".
  # Treat missing values as non-smokers. 81% of all non-missing values are no so it's by far the most likely value.
  # Understood that this is a very imperfect rationale.
  # Encode the final set of unique values (no, sometimes and yes) to integers between 0 and 5, ordered and scaled based
  # on the frequency of smoking.
  # Stretch the scale out quite large to place a higher value on similarity, as non-smokers likely value high similarity
  # on this dimension.
  'smokes',

  # speaks:
  # String
  # 50 missing values (~0%)
  # 7,643 unique values: english, etc.
  # Consolidate similar values to the 9 most common ones to reduce unique values, and trim to one language each,
  # prioritizing the first mentioned one, on the (I believe correct) assumption that it's their strongest one.
  # Later on if might be nice to allow for the model to consider multiple languages, though.
  # Replace missing values with english, as this is a US data set and almost all in the dataset speak it (either
  # primarily or secondarily).
  # Don't encode, pass this through exactly as we are going to use this for a direct value look-up.
  'speaks'
]

# Future feature engineering work:
#
# Consider replacing the following fields which are currently one-hot encoded into continuous values between 0 and 1.
# * offspring (maybe split into two)
#
# Features that will likely still be one-hot encoded and need to think about:
# * ethnicity
# * religion

# Input data columns not using:
# * essay0, essay1, essay2, essay3, essay4, essay5, essay6, essay7, essay8, essay9: Skip these for now, but intend to
#   use them later, possibly by layering some NLP similarity on top of the more basic features.
# * last_online, sign: Not useful.
# * income: Would be interesting, but it's apparently an optional field: 81% of all rows have a value of -1. Not enough
#   to be useful.
# * height: Not really relevant for matchmaking. Is also going to correlate with sex and create noise in that sense.
#   *Could* be untangled, as an academic exercise, but again, it's not relevant enough.
# * location: The population of this dataset is entirely in and around San Francisco, California, US. There are 199
#   unique values, such as "san francisco, california". Going to drop this and pretend that it's a local dataset. Not
#   interested in doing anything regarding location matching/radius/etc for the purposes of this project.
# * job: This is more like industry (e.g. entertainment, banking, etc). There are also a lot of missing "other", etc
#   values. Not particularly relevant for this purpose; lawyers aren't necessarily looking to date other lawyers. Might
#   do something with this later on though.

DIRECT_LOOKUP_FEATURES = ['sex', 'sexual_orientation', 'speaks']

# The wider the range, the more stretched out the scale, the greater the distance of variations, the less near/similar
# variations are, the less likely they are to be a near neighbor. Therefore, use a wider range for continuous features
# that should match more exactly, as differences will have a larger influence on similarity. And use a smaller range for
# those that should allow more variation and have a smaller influence on similarity. Sounds counter-intuitive, but
# reasons out.
#
# By the way, these are 100% arbitrary based on how important I think these features are to people when dating, e.g. how
# important is it to a vegan that their partners are also (or close to) vegan?
CONTINUOUS_FEATURE_AGE_SCALER = sklearn.preprocessing.MinMaxScaler(feature_range = (0, 5))
CONTINUOUS_FEATURE_BODY_TYPE_SCALER = sklearn.preprocessing.MinMaxScaler(feature_range = (0, 0.4))
CONTINUOUS_FEATURE_DIET_SCALER = sklearn.preprocessing.MinMaxScaler(feature_range = (0, 1.5))
CONTINUOUS_FEATURE_DRINKS_SCALER = sklearn.preprocessing.MinMaxScaler(feature_range = (0, 0.7))
CONTINUOUS_FEATURE_DRUGS_SCALER = sklearn.preprocessing.MinMaxScaler(feature_range = (0, 3))
CONTINUOUS_FEATURE_EDUCATION_SCALER = sklearn.preprocessing.MinMaxScaler(feature_range = (0, 0.5))
CONTINUOUS_FEATURE_SMOKES_SCALER = sklearn.preprocessing.MinMaxScaler(feature_range = (0, 3))

CATEGORICAL_FEATURES_TO_ONE_HOT_ENCODE = [
  'ethnicity',
  'offspring',
  'religion'
]

# The OrdinalEncoder assigns integers from 0 to n-1 based on the number of categories. However, not all categories have
# an equal scale between values. For example, there is a bigger conceptual gap between not smoking and sometimes
# smoking, than between sometimes smoking and regularly smoking. Therefore introduce buffer values in the categories to
# pad out the resulting scale. For smoking, this means that the values for no, sometimes and yes will be 0, 3 and 5
# instead of 0, 1 and 2. Is there a more standard way of doing this? Maybe, but not that I could find.
CATEGORICAL_FEATURES_TO_ORDINAL_ENCODE = ['body_type', 'diet', 'drinks', 'drugs', 'education', 'smokes']

CATEGORICAL_FEATURE_BODY_TYPE_ORDINALITIES = ['thin', 'fit', 'average', 'curvy', 'buffer1', 'overweight']
CATEGORICAL_FEATURE_DIET_ORDINALITIES = ['vegan', 'vegetarian', 'buffer1', 'buffer2', 'anything']
CATEGORICAL_FEATURE_DRINKS_ORDINALITIES = ['never', 'buffer1', 'rarely', 'socially', 'buffer2', 'often']
CATEGORICAL_FEATURE_DRUGS_ORDINALITIES = ['never', 'buffer1', 'sometimes', 'buffer2', 'often']
CATEGORICAL_FEATURE_EDUCATION_ORDINALITIES = ['less_than_high_school', 'high_school', 'in_progress_study', 'completed_undergraduate_study', 'completed_postgraduate_study']
CATEGORICAL_FEATURE_SMOKES_ORDINALITIES = ['no', 'buffer1', 'buffer2', 'sometimes', 'buffer4', 'yes']

CATEGORICAL_FEATURES_ORDINAL_ENCODER = sklearn.preprocessing.OrdinalEncoder(
  categories = [
    CATEGORICAL_FEATURE_BODY_TYPE_ORDINALITIES,
    CATEGORICAL_FEATURE_DIET_ORDINALITIES,
    CATEGORICAL_FEATURE_DRINKS_ORDINALITIES,
    CATEGORICAL_FEATURE_DRUGS_ORDINALITIES,
    CATEGORICAL_FEATURE_EDUCATION_ORDINALITIES,
    CATEGORICAL_FEATURE_SMOKES_ORDINALITIES
  ],
  dtype = int
)

# Encode (and scale) categorical features (such as cats yes/no) into a binary 0 or positive number value (depending on
# scaling).
CATEGORICAL_FEATURE_PETS_CATS_LABEL_ENCODINGS = { False: 0, True: 0.1 }
CATEGORICAL_FEATURE_PETS_DOGS_LABEL_ENCODINGS = { False: 0, True: 0.1 }

# Preserve the order of the features from the input data, after both applying and reversing one-hot encoding.
# Note that the one-hot encoded features need to match the column name(s) both with and without the encoding.
FEATURE_SORT_ORDER = [
  r'^age$',
  r'^sex$',
  r'^sexual_orientation$',
  r'^body_type$',
  r'^diet$',
  r'^drinks$',
  r'^drugs$',
  r'^education$',
  r'^ethnicity(?:_.*)?$',
  r'^offspring(?:_.*)?$',
  r'^pets_cats$',
  r'^pets_dogs$',
  r'^religion(?:_.*)?$',
  r'^smokes$',
  r'^speaks$'
]

def load_input_data():
  return pd.read_csv(
    INPUT_DATA_PATH,
    header = 0,
    names = INPUT_DATA_COLUMN_NAMES,
    usecols = INPUT_DATA_COLUMNS_TO_USE
  )

def preprocess_input_data(data_frame):
  data_frame = filter_and_drop_relationship_status(data_frame)
  data_frame = consolidate_values(data_frame)
  data_frame = split_and_drop_pets(data_frame)

  # Apply one-hot encoding to categorical features.
  data_frame = pd.get_dummies(
    data_frame,
    columns = CATEGORICAL_FEATURES_TO_ONE_HOT_ENCODE,
    sparse = False
  )

  # Apply ordinal/positional encodings to categorical features.
  data_frame[CATEGORICAL_FEATURES_TO_ORDINAL_ENCODE] = CATEGORICAL_FEATURES_ORDINAL_ENCODER.fit_transform(
    data_frame[CATEGORICAL_FEATURES_TO_ORDINAL_ENCODE].to_numpy()
  )

  # Apply label replacement encodings to categorical features.
  data_frame = data_frame.replace(
    {
      'pets_cats': CATEGORICAL_FEATURE_PETS_CATS_LABEL_ENCODINGS,
      'pets_dogs': CATEGORICAL_FEATURE_PETS_DOGS_LABEL_ENCODINGS
    }
  )

  # Linearly scale/normalize continuous features.
  data_frame[['age']] = CONTINUOUS_FEATURE_AGE_SCALER.fit_transform(data_frame[['age']].to_numpy())
  data_frame[['body_type']] = CONTINUOUS_FEATURE_BODY_TYPE_SCALER.fit_transform(data_frame[['body_type']].to_numpy())
  data_frame[['diet']] = CONTINUOUS_FEATURE_DIET_SCALER.fit_transform(data_frame[['diet']].to_numpy())
  data_frame[['drinks']] = CONTINUOUS_FEATURE_DRINKS_SCALER.fit_transform(data_frame[['drinks']].to_numpy())
  data_frame[['drugs']] = CONTINUOUS_FEATURE_DRUGS_SCALER.fit_transform(data_frame[['drugs']].to_numpy())
  data_frame[['education']] = CONTINUOUS_FEATURE_EDUCATION_SCALER.fit_transform(data_frame[['education']].to_numpy())
  data_frame[['smokes']] = CONTINUOUS_FEATURE_SMOKES_SCALER.fit_transform(data_frame[['smokes']].to_numpy())

  data_frame = Utilities.sort_data_frame(data_frame)

  return data_frame

# Drop rows where relationship_status is unknown.
# Also drop rows that are seeing someone or married. Get off OkCupid.
# Then drop the column, as it is no longer needed outside preprocessing.
def filter_and_drop_relationship_status(data_frame):
  relationship_statuses_to_drop = ['unknown', 'seeing someone', 'married']
  data_frame = data_frame.drop(data_frame[data_frame['relationship_status'].isin(relationship_statuses_to_drop)].index)
  data_frame.drop('relationship_status', axis = 1, inplace = True)

  return data_frame

# Per-feature value consolidation. Label encoding is a two-step process:
# * First, consolidate a wide range of values to a smaller set to reduce the number of unique ones, e.g. "full figured"
#   to "curvy" and "skinny" to "thin". This is irreversible.
# * Next, in a later method, apply encodings using Scikit's encoders to turn everything into machine-usable integers.
#   This second step is reversible.
def consolidate_values(data_frame):
  # First do standard string replacements.
  data_frame = data_frame.replace(
    {
      'body_type': {
        'athletic': 'fit',
        'skinny': 'thin',
        'jacked': 'fit',
        'full figured': 'curvy',
        'a little extra': 'curvy',
        'rather not say': 'average',
        'used up': 'average', # I don't even know what this means...
        np.nan: 'average'
      },
      'diet': {
        np.nan: 'anything'
      },
      'drinks': {
        'very often': 'often',
        'not at all': 'never',
        'desperately': 'often', # I presume this means often.
        np.nan: 'socially'
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
        'dropped out of space camp': 'high_school',
        'working on space camp': 'high_school',
        'space camp': 'high_school',
        'graduated from space camp': 'high_school',
        np.nan: 'high_school'
      },
      'ethnicity': {
        'hispanic / latin': 'hispanic_latin',
        np.nan: 'unknown',
      },
      'smokes': {
        'when drinking': 'sometimes',
        'trying to quit': 'sometimes',
        np.nan: 'no'
      },
      'speaks': {
        np.nan: 'english'
      },
      'offspring': {
        'wants kids': 'no_kids',
        'might want kids': 'no_kids',
        'doesn\'t have kids': 'no_kids',
        'doesn\'t have kids, but might want them': 'no_kids',
        'doesn\'t have kids, but wants them': 'no_kids',
        'doesn\'t want kids': 'no_kids_dont_want_any',
        'doesn\'t have kids, and doesn\'t want any': 'no_kids_dont_want_any',
        'has a kid': 'has_kids',
        'has a kid, but doesn\'t want more': 'has_kids_but_no_more',
        'has a kid, and might want more': 'has_kids',
        'has a kid, and wants more': 'has_kids',
        'has kids': 'has_kids',
        'has kids, but doesn\'t want more': 'has_kids_but_no_more',
        'has kids, and might want more': 'has_kids',
        'has kids, and wants more': 'has_kids',
        np.nan: 'unknown'
      },
      'pets': {
        'dislikes dogs and dislikes cats': '',
        'dislikes cats': '',
        'dislikes dogs': '',
        'dislikes dogs and likes cats': '',
        'likes cats': '',
        'likes dogs': '',
        'likes dogs and dislikes cats': '',
        'likes dogs and likes cats': '',
        'has cats': 'cats',
        'dislikes dogs and has cats': 'cats',
        'likes dogs and has cats': 'cats',
        'has dogs': 'dogs',
        'has dogs and dislikes cats': 'dogs',
        'has dogs and likes cats': 'dogs',
        'has dogs and has cats': 'cats,dogs',
        np.nan: ''
      },
      # Controversial...
      'religion': {
        'atheism but not too serious about it': 'atheism',
        'atheism and somewhat serious about it': 'atheism',
        'atheism and very serious about it': 'atheism',
        'atheism and laughing about it': 'atheism',
        'agnosticism but not too serious about it': 'agnosticism',
        'agnosticism and somewhat serious about it': 'agnosticism',
        'agnosticism and very serious about it': 'agnosticism', # What does it even mean to be a very serious agnostic?
        'agnosticism and laughing about it': 'agnosticism',
        'buddhism but not too serious about it': 'buddhism',
        'buddhism and somewhat serious about it': 'buddhism',
        'buddhism and very serious about it': 'buddhism',
        'buddhism and laughing about it': 'buddhism',
        'hinduism but not too serious about it': 'hinduism',
        'hinduism and somewhat serious about it': 'hinduism',
        'hinduism and very serious about it': 'hinduism',
        'hinduism and laughing about it': 'hinduism',
        'islam but not too serious about it': 'islam',
        'islam and somewhat serious about it': 'islam',
        'islam and very serious about it': 'islam',
        'islam and laughing about it': 'islam',
        'judaism but not too serious about it': 'judaism',
        'judaism and somewhat serious about it': 'judaism',
        'judaism and very serious about it': 'judaism',
        'judaism and laughing about it': 'judaism',
        'christianity but not too serious about it': 'christianity',
        'christianity and somewhat serious about it': 'christianity',
        'christianity and very serious about it': 'christianity',
        'christianity and laughing about it': 'christianity',
        'catholicism but not too serious about it': 'catholicism',
        'catholicism and somewhat serious about it': 'catholicism',
        'catholicism and very serious about it': 'catholicism',
        'catholicism and laughing about it': 'catholicism',
        'other but not too serious about it': 'other',
        'other and somewhat serious about it': 'other',
        'other and very serious about it': 'other',
        'other and laughing about it': 'other',
        np.nan: 'unknown'
      }
    }
  )

  # Next do regular expression value consolidations.
  data_frame = data_frame.replace(
    {
      # Filter everything down to vegan, vegetarian or other. See above for notes about reigious diets (e.g. kosher).
      'diet': {
        r'.*vegan.*': 'vegan',
        r'.*vegetarian.*': 'vegetarian',
        r'^((?!vegan|vegetarian).)*$': 'anything'
      },
      # I am *not* mapping 217 unique values!
      'ethnicity': {
        r'^(?!white$|asian$|hispanic_latin$|black$).*': 'unknown'
      },
      # Believe it or not, this is enough to consolidate and trim down to each row having one normalized language
      # (favouring their first listed).
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

  return data_frame

def split_and_drop_pets(data_frame):
  data_frame['pets_cats'] = data_frame['pets'].str.contains('cats')
  data_frame['pets_dogs'] = data_frame['pets'].str.contains('dogs')

  data_frame.drop('pets', axis = 1, inplace = True)

  return data_frame
