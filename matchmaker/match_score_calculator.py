import math
import scipy

from . import data_preprocessing as DataPreprocessing

# Returns a list of match scores for each match (compared to the input).
def calculate_match_score(input_data_frame, matches_data_frame):
  input_data_frame = input_data_frame.drop(columns = DataPreprocessing.DIRECT_LOOKUP_FEATURES)
  matches_data_frame = matches_data_frame.drop(columns = DataPreprocessing.DIRECT_LOOKUP_FEATURES)

  input_data = input_data_frame.iloc[0].values

  match_scores = []

  for index, match_data in matches_data_frame.iterrows():
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.braycurtis.html#scipy.spatial.distance.braycurtis
    # The distance is a decimal between 0 and 1, with 0 being identical and increasing towards 1 with difference.
    # From documentation: "The Bray-Curtis distance is in the range [0, 1] if all coordinates are positive..."
    # Flip it so that the higher the score the more similar it is, and convert to a rounded-down percentage.
    distance = scipy.spatial.distance.braycurtis(input_data.astype(float), match_data.values.astype(float))
    match_score = math.floor((1 - distance) * 100)
    match_scores.append(match_score)

  return match_scores
