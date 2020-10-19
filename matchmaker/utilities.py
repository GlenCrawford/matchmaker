from . import data_preprocessing as DataPreprocessing
import pandas as pd

def reverse_preprocessing(data_frame):
  data_frame = reverse_continuous_scaling(data_frame)
  data_frame = reverse_ordinal_encoding(data_frame)
  data_frame = reverse_one_hot_encoding(data_frame)

  return data_frame

# For each set of columns that were derived from a feature that was one-hot encoded during preprocessing, consolidate
# back to a single column. E.g. columns body_type_thin and body_type_fit get merged into one column called body_type
# with values thin and fit.
def reverse_one_hot_encoding(data_frame):
  for one_hot_encoded_feature in DataPreprocessing.CATEGORICAL_FEATURES_TO_ONE_HOT_ENCODE:
    # Get all the columns dervied from this feature.
    columns_for_feature = [column for column in data_frame.columns if column.startswith(one_hot_encoded_feature)]

    # Consolidate the columns into one.
    data_frame[one_hot_encoded_feature] = data_frame[columns_for_feature].idxmax(1)

    # Remove the feature name prefix.
    data_frame[one_hot_encoded_feature] = data_frame[one_hot_encoded_feature].str.replace(f'{one_hot_encoded_feature}_', '')

    # Drop all the derived columns.
    data_frame.drop(columns = columns_for_feature, inplace = True)

  data_frame = sort_data_frame(data_frame)

  return data_frame

# Use the scalers (which have their range and fitting stored within them) to reverse the scaling performed during the
# preprocessing step.
def reverse_continuous_scaling(data_frame):
  data_frame[['age']] = DataPreprocessing.CONTINUOUS_FEATURE_AGE_SCALER.inverse_transform(data_frame[['age']].to_numpy())
  data_frame[['body_type']] = DataPreprocessing.CONTINUOUS_FEATURE_BODY_TYPE_SCALER.inverse_transform(data_frame[['body_type']].to_numpy())
  data_frame[['drinks']] = DataPreprocessing.CONTINUOUS_FEATURE_DRINKS_SCALER.inverse_transform(data_frame[['drinks']].to_numpy())
  data_frame[['drugs']] = DataPreprocessing.CONTINUOUS_FEATURE_DRUGS_SCALER.inverse_transform(data_frame[['drugs']].to_numpy())
  data_frame[['education']] = DataPreprocessing.CONTINUOUS_FEATURE_EDUCATION_SCALER.inverse_transform(data_frame[['education']].to_numpy())
  data_frame[['smokes']] = DataPreprocessing.CONTINUOUS_FEATURE_SMOKES_SCALER.inverse_transform(data_frame[['smokes']].to_numpy())

  return data_frame

# Use the categorical features ordinal encoders (which still has its categories stored from preprocessing) to reverse
# the encodings back to the original category labels.
def reverse_ordinal_encoding(data_frame):
  data_frame[DataPreprocessing.CATEGORICAL_FEATURES_TO_ORDINAL_ENCODE] = DataPreprocessing.CATEGORICAL_FEATURES_ORDINAL_ENCODER.inverse_transform(data_frame[DataPreprocessing.CATEGORICAL_FEATURES_TO_ORDINAL_ENCODE].to_numpy())

  return data_frame

# Apply the specified regular expression-based sort order to the columns in the data frame.
def sort_data_frame(data_frame):
  return pd.concat([data_frame.filter(regex = regex) for regex in DataPreprocessing.FEATURE_SORT_ORDER], axis = 1)
