from matchmaker import *
from sklearn.neighbors import NearestNeighbors
from scipy import stats

NEAREST_NEIGHBORS_TO_RETRIEVE = 40 # Just because that's what fits on my screen ;)

def main():
  population_data_frame = DataPreprocessing.load_input_data()
  population_data_frame = add_input_data_to_population(population_data_frame)
  population_data_frame = DataPreprocessing.preprocess_input_data(population_data_frame)

  # Now they have been preprocessed together, extract out the input row as a data frame.
  input_data_frame = population_data_frame.loc[['input']]
  population_data_frame.drop('input', inplace = True)

  # print(population_data_frame['speaks'].value_counts(dropna = False))
  # print(population_data_frame.filter(regex = r'^drinks(.*)$', axis = 1))
  #
  # exit()

  population_data_frame = apply_direct_lookups(input_data_frame, population_data_frame)

  # If there is no population to search for similarity within after applying the direct lookups, stop here.
  if len(population_data_frame) == 0:
    print('No results :(')
    exit()

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

  all_distances, all_indices = nearest_neighbors_model.kneighbors(
    input_data_frame.loc[:, ~input_data_frame.columns.isin(DataPreprocessing.DIRECT_LOOKUP_FEATURES)]
  )

  # We did inference for all rows, now extract out the X number of ones that we want. The returned indices and distances
  # are sorted by distance in ascending order, so nearest first.
  nearest_neighbors_distances = [all_distances[0][:NEAREST_NEIGHBORS_TO_RETRIEVE]]
  nearest_neighbors_indices = [all_indices[0][:NEAREST_NEIGHBORS_TO_RETRIEVE]]

  # Reverse some of the data preprocessing to make the data frame prettier for output.
  population_data_frame = Utilities.reverse_one_hot_encoding(population_data_frame)
  input_data_frame = Utilities.reverse_one_hot_encoding(input_data_frame)

  population_data_frame = Utilities.reverse_continuous_scaling(population_data_frame)
  input_data_frame = Utilities.reverse_continuous_scaling(input_data_frame)

  population_data_frame = Utilities.reverse_ordinal_encoding(population_data_frame)
  input_data_frame = Utilities.reverse_ordinal_encoding(input_data_frame)

  # Calculate a "similarity score" of each neighbor. This is not the absolute similarity of the neighbor from the target
  # one, it's more of its percentile ranking within the distances of all rows, meaning that it's effectively its ranking
  # within the population, converted to a percentage/score.
  nearest_neighbors_similarity_score = [
    round((100 - stats.percentileofscore(all_distances[0], distance, 'rank')), 2)
    for distance in nearest_neighbors_distances[0]
  ]

  # Fetch the neighbors' from the population data frame.
  nearest_neighbors_data_frame = population_data_frame.iloc[nearest_neighbors_indices[0]]

  # Zip the similarity scores into the nearest neighbors as the first column.
  nearest_neighbors_data_frame.insert(loc = 0, column = 'score', value = nearest_neighbors_similarity_score)

  print('Input data:\n', input_data_frame, '\n\n')
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
    'likes dogs and dislikes cats', # pets 🚫🐈
    'christianity', # religion
    'no', # smokes
    'english' # speaks
  ]

  # Override single value for testing/debugging:
  # population_data_frame.at['input', 'speaks'] = 'french'

  return population_data_frame

# Apply pure logic-based filters to the population for features that must be exact, not merely similar.
def apply_direct_lookups(input_data_frame, population_data_frame):
  input_sex = input_data_frame.iloc[0]['sex']
  input_sexual_orientation = input_data_frame.iloc[0]['sexual_orientation']
  input_speaks = input_data_frame.iloc[0]['speaks']

  # If input is straight:
  # * Filter sex to other sex.
  # * Filter sexual_orientation to straight and bisexual.
  if input_sexual_orientation == 'straight':
    population_data_frame = population_data_frame[
      population_data_frame['sex'].isin([{ 'm': 'f', 'f': 'm' }[input_sex]]) &
      population_data_frame['sexual_orientation'].isin(['straight', 'bisexual'])
    ]
  # If input is gay:
  # * Filter sex to input sex.
  # * Filter sexual_orientation to gay and bisexual.
  elif input_sexual_orientation == 'gay':
    population_data_frame = population_data_frame[
      (population_data_frame['sex'] == input_sex)
      &
      population_data_frame['sexual_orientation'].isin(['gay', 'bisexual'])
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
      population_data_frame = population_data_frame[
        ((population_data_frame['sex'] == 'm') & population_data_frame['sexual_orientation'].isin(['gay', 'bisexual']))
        |
        ((population_data_frame['sex'] == 'f') & population_data_frame['sexual_orientation'].isin(['straight', 'bisexual']))
      ]
    elif input_sex == 'f':
      population_data_frame = population_data_frame[
        ((population_data_frame['sex'] == 'm') & population_data_frame['sexual_orientation'].isin(['straight', 'bisexual']))
        |
        ((population_data_frame['sex'] == 'f') & population_data_frame['sexual_orientation'].isin(['gay', 'bisexual']))
      ]

  # Filter by the input language.
  population_data_frame = population_data_frame[population_data_frame['speaks'] == input_speaks]

  return population_data_frame

if __name__ == '__main__':
  main()
