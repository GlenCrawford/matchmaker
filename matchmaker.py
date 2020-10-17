from matchmaker import *
from sklearn.neighbors import NearestNeighbors
from scipy import stats

def main():
  population_data_frame = DataPreprocessing.load_input_data()
  population_data_frame = add_input_data_to_population(population_data_frame)
  population_data_frame = DataPreprocessing.preprocess_input_data(population_data_frame)

  # Now they have been preprocessed together, extract out the input row as a data frame.
  input_data_frame = population_data_frame.loc[['input']]
  population_data_frame.drop('input', inplace = True)

  # Get X most similar rows (nearest neighbors).
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

  # Can these both be done at once?
  all_distances, all_indices = nearest_neighbors_model.kneighbors(
    input_data_frame.loc[:, ~input_data_frame.columns.isin(DataPreprocessing.DIRECT_LOOKUP_FEATURES)]
  )

  nearest_neighbors_distances, nearest_neighbors_indices = nearest_neighbors_model.kneighbors(
    input_data_frame.loc[:, ~input_data_frame.columns.isin(DataPreprocessing.DIRECT_LOOKUP_FEATURES)],
    n_neighbors = 5
  )

  population_data_frame = Utilities.reverse_one_hot_encoding(population_data_frame)
  input_data_frame = Utilities.reverse_one_hot_encoding(input_data_frame)

  population_data_frame = Utilities.reverse_continuous_scaling(population_data_frame)
  input_data_frame = Utilities.reverse_continuous_scaling(input_data_frame)

  # Calculate a "similarity score" of each neighbor. This is not the absolute similarity of the neighbor from the target
  # one, it's more of its percentile ranking within the distances of all rows, meaning that it's effectively its ranking
  # within the population, converted to a percentage/score.
  nearest_neighbors_similarity_score = [
    100 - stats.percentileofscore(all_distances[0], distance, 'rank')
    for distance in nearest_neighbors_distances[0]
  ]

  # Fetch the neighbors' from the population data frame.
  nearest_neighbors_data_frame = population_data_frame.iloc[nearest_neighbors_indices[0]]

  print('Input data:\n', input_data_frame, '\n\n')
  print('All distances (closer to zero the better):\n', all_distances[0], '\n\n')
  print('Nearest Neighbors indices (within the results, not initial data set):\n', nearest_neighbors_indices[0], '\n\n')
  print('Nearest Neighbors distances (closer to zero the better):\n', nearest_neighbors_distances[0], '\n\n')
  print('Nearest Neighbors similarity score (higher the better):\n', nearest_neighbors_similarity_score, '\n\n')
  print('Nearest Neighbors:\n', nearest_neighbors_data_frame)

def add_input_data_to_population(population_data_frame):
  # Only specify values for features in INPUT_DATA_COLUMNS_TO_USE, in the order in which they appear in the input data.
  population_data_frame.loc['input'] = [
    '30', # age
    'single', # relationship_status
    'm', # sex
    'straight', # sexual_orientation
    'thin', # body_type
    'anything', # diet
    'rarely', # drinks
    'never', # drugs
    'graduated from college/university', # education
    'white', # ethnicity
    'doesn\'t have kids, but wants them', # offspring
    'likes dogs and dislikes cats', # pets
    'christianity', # religion
    'no', # smokes
    'english' # speaks
  ]

  return population_data_frame

if __name__ == '__main__':
  main()
