import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy import stats

from . import data_preprocessing as DataPreprocessing
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

    # By default, return the distances of all rows. This will be quite inefficient, so override this when looking for a
    # smaller subset by specifying a smaller number when invoking .kneighbors.
    # A formula of "minkowski" and p of 2 makes for a Euclidean distance metric.
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
    nearest_neighbors_model = NearestNeighbors(
      n_neighbors = len(population_data_frame),
      algorithm = 'auto',
      metric = 'minkowski',
      p = 2
    ).fit(population_data_frame.loc[:, ~population_data_frame.columns.isin(DataPreprocessing.DIRECT_LOOKUP_FEATURES)])

    Serialization.save_model(nearest_neighbors_model)

  # Convert the input data vector into a dataframe and pre-process, using pre-fitted encoders.
  input_data.pop(1) # Remove relationship_status.
  input_data_frame = pd.DataFrame([input_data], columns = candidates_data_frame.columns)
  input_data_frame = DataPreprocessing.preprocess_input_data(input_data_frame, use_fitted_encoders = True)

  # Get the similarity/distances of the entire population for the input.
  population_distances, population_indices = nearest_neighbors_model.kneighbors(
    input_data_frame.loc[:, ~input_data_frame.columns.isin(DataPreprocessing.DIRECT_LOOKUP_FEATURES)]
  )

  population_indices = population_indices[0]
  population_distances = population_distances[0]

  # We did inference for the entire population, now we need to reduce that down to only the candidates and filter down
  # to the X number of results that we want.
  #
  # Note that population_indices are the indices (not labels) of the row in the population data frame on which the model
  # was fitted.
  #
  # The returned indices and distances are sorted by distance in ascending order, so nearest first.

  # Reduce the population indices and distances down to only those that are candidates.
  candidates_indices = []
  candidates_distances = []

  for index, population_index in enumerate(population_indices):
    population_distance = population_distances[index]

    population_label = population_data_frame.index[population_index]

    if population_label in candidates_data_frame.index:
      candidates_indices.append(population_index)
      candidates_distances.append(population_distance)

  # Slice out only the desired number of candidates.
  nearest_neighbors_indices = candidates_indices[:matches_to_retrieve]
  nearest_neighbors_distances = candidates_distances[:matches_to_retrieve]

  # Calculate a "similarity score" of each neighbor. This is not the absolute similarity of the neighbor from the target
  # one, it's more of its percentile ranking within the distances of all candidates, meaning that it's effectively its
  # ranking within the candidates, converted to a percentage/score.
  nearest_neighbors_similarity_score = [
    round((100 - stats.percentileofscore(candidates_distances, distance, 'rank')), 2)
    for distance in nearest_neighbors_distances
  ]

  # Fetch the neighbors' rows from the population data frame.
  nearest_neighbors_data_frame = population_data_frame.iloc[nearest_neighbors_indices].copy()

  # If we didn't fit the model, then the population never got preprocessed. But we still want to do value consolidation
  # and things for display purposes. So preprocess just the nearest neighbour results.
  if not train_model:
    nearest_neighbors_data_frame = DataPreprocessing.preprocess_input_data(nearest_neighbors_data_frame, use_fitted_encoders = True)

  # Reverse some of the data preprocessing to make the data frames prettier for output.
  input_data_frame, nearest_neighbors_data_frame = map(
    Utilities.reverse_preprocessing,
    (input_data_frame, nearest_neighbors_data_frame)
  )

  # Zip the similarity scores into the nearest neighbors as the first column.
  nearest_neighbors_data_frame.insert(loc = 0, column = 'score', value = nearest_neighbors_similarity_score)

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
