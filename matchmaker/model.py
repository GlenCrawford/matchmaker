import pandas as pd
from sklearn.neighbors import NearestNeighbors

from . import data_preprocessing as DataPreprocessing
from . import match_score_calculator as MatchScoreCalculator
from . import serialization as Serialization
from . import utilities as Utilities

def execute(input_data, force_training, matches_to_retrieve):
  population_data_frame = DataPreprocessing.load_input_data()

  candidates_data_frame = apply_direct_lookups(input_data, population_data_frame)

  # If there are no candidates to search for similarity within after applying the direct lookups, stop here.
  if len(candidates_data_frame) == 0:
    return input_data, []

  nearest_neighbors_model = Serialization.load_model()

  # If there is no pre-trained model, always train a new one. If there is, use it unless forced to re-train a new one.
  train_model = (nearest_neighbors_model is None) or force_training
  if train_model:
    population_data_frame = DataPreprocessing.preprocess_input_data(population_data_frame, use_fitted_encoders = False)

    # Fit and save the model, trained with the entire population.
    # A formula of "minkowski" and p of 2 makes for a Euclidean distance metric.
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
    nearest_neighbors_model = NearestNeighbors(
      algorithm = 'auto',
      metric = 'minkowski',
      p = 2
    ).fit(population_data_frame.loc[:, ~population_data_frame.columns.isin(DataPreprocessing.DIRECT_LOOKUP_FEATURES)])

    Serialization.save_model(nearest_neighbors_model)

  # Convert the input data vector into a dataframe and pre-process, using pre-fitted encoders.
  input_data.pop(1) # Remove relationship_status.
  input_data_frame = pd.DataFrame([input_data], columns = candidates_data_frame.columns)
  input_data_frame = DataPreprocessing.preprocess_input_data(input_data_frame, use_fitted_encoders = True)

  # Fetch the indices of the nearest neighbors to the input. This is among the entire population, which will need to be
  # filtered down to candidates, so get more than we need to ensure there are enough to get the amount that we want.
  population_indices = nearest_neighbors_model.kneighbors(
    input_data_frame.loc[:, ~input_data_frame.columns.isin(DataPreprocessing.DIRECT_LOOKUP_FEATURES)],
    n_neighbors = (matches_to_retrieve * 5),
    return_distance = False
  )[0]

  # We did inference among the entire population, now we need to reduce that down to only the candidates and filter down
  # to the X number of results that we want.
  #
  # Note that population_indices are the indices (not labels) of the row in the population data frame on which the model
  # was fitted.
  #
  # The returned indices are sorted by distance in ascending order, so nearest first.

  nearest_neighbors_indices = []

  for index, population_index in enumerate(population_indices):
    population_label = population_data_frame.index[population_index]

    if population_label in candidates_data_frame.index:
      nearest_neighbors_indices.append(population_index)

      # Stop once we have enough.
      if len(nearest_neighbors_indices) >= matches_to_retrieve:
        break

  # Fetch the neighbors' rows from the population data frame.
  nearest_neighbors_data_frame = population_data_frame.iloc[nearest_neighbors_indices].copy()

  # If we didn't fit the model, then the population never got preprocessed. But we still want to do value consolidation
  # and things for display purposes. So preprocess just the nearest neighbour results.
  if not train_model:
    nearest_neighbors_data_frame = DataPreprocessing.preprocess_input_data(nearest_neighbors_data_frame, use_fitted_encoders = True)

  nearest_neighbors_match_scores = MatchScoreCalculator.calculate_match_score(input_data_frame, nearest_neighbors_data_frame)

  # Reverse some of the data preprocessing to make the data frames prettier for output.
  input_data_frame, nearest_neighbors_data_frame = map(
    Utilities.reverse_preprocessing,
    (input_data_frame, nearest_neighbors_data_frame)
  )

  # Zip the match scores into the nearest neighbors as the first column and sort by them.
  nearest_neighbors_data_frame.insert(loc = 0, column = 'score', value = nearest_neighbors_match_scores)
  nearest_neighbors_data_frame.sort_values(by = 'score', ascending = False, inplace = True, ignore_index = True)

  return input_data_frame, nearest_neighbors_data_frame

# Apply pure logic-based filters to the population for features that must be exact, not merely similar.
def apply_direct_lookups(input_data, population_data_frame):
  input_sex = input_data[2]
  input_sexual_orientation = input_data[3]
  input_speaks = input_data[14]

  candidates_data_frame = population_data_frame

  # If input is straight:
  # * Filter sex to other sex.
  # * Filter sexual_orientation to straight and bisexual.
  if input_sexual_orientation == 'straight':
    candidates_data_frame = candidates_data_frame[
      candidates_data_frame['sex'].isin([{ 'm': 'f', 'f': 'm' }[input_sex]]) &
      candidates_data_frame['sexual_orientation'].isin(['straight', 'bisexual'])
    ]
  # If input is gay:
  # * Filter sex to input sex.
  # * Filter sexual_orientation to gay and bisexual.
  elif input_sexual_orientation == 'gay':
    candidates_data_frame = candidates_data_frame[
      (candidates_data_frame['sex'] == input_sex)
      &
      candidates_data_frame['sexual_orientation'].isin(['gay', 'bisexual'])
    ]
  # If input is bisexual:
  # * If input is male, filter:
  #   - (sex is male) and (sexual_orientation is gay or bisexual)
  #   - or
  #   - (sex is female) and (sexual_orientation is straight or bisexual)
  # * If input is female, filter:
  #   - (sex is male) and (sexual_orientation is straight or bisexual)
  #   - or
  #   - (sex is female) and (sexual_orientation is gay or bisexual)
  elif input_sexual_orientation == 'bisexual':
    if input_sex == 'm':
      candidates_data_frame = candidates_data_frame[
        ((candidates_data_frame['sex'] == 'm') & candidates_data_frame['sexual_orientation'].isin(['gay', 'bisexual']))
        |
        ((candidates_data_frame['sex'] == 'f') & candidates_data_frame['sexual_orientation'].isin(['straight', 'bisexual']))
      ]
    elif input_sex == 'f':
      candidates_data_frame = candidates_data_frame[
        ((candidates_data_frame['sex'] == 'm') & candidates_data_frame['sexual_orientation'].isin(['straight', 'bisexual']))
        |
        ((candidates_data_frame['sex'] == 'f') & candidates_data_frame['sexual_orientation'].isin(['gay', 'bisexual']))
      ]

  # Filter by the input language. This is a comma-separated list of languages; filter down to any row that includes the
  # input language, meaning there will be at least one language in common.
  candidates_data_frame = candidates_data_frame[candidates_data_frame['speaks'].str.contains(input_speaks)]

  return candidates_data_frame
