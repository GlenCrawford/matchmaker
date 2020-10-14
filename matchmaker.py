from matchmaker import *
from sklearn.neighbors import NearestNeighbors
from scipy import stats

def main():
  data_frame = DataPreprocessing.load_input_data()
  data_frame = DataPreprocessing.preprocess_input_data(data_frame)

  # For inspecting columns while preprocessing.
  # print(data_frame['speaks'].describe())
  # print(data_frame['speaks'].value_counts(dropna = False))
  # print(data_frame.filter(regex = r'^speaks(.*)$', axis = 1))
  # exit()

  # Quick proof-of-concept testing of scikit-learn KNN.
  # Remove the columns that we later plan on using as exact look-ups.
  data_frame.drop(columns = ['relationship_status', 'sex', 'sexual_orientation'], inplace = True)

  # Subtract out a random row to test with.
  random_row = data_frame.sample()
  index_of_random_row = random_row.iloc[[0]].index.item()
  data_frame.drop(index_of_random_row, inplace = True)

  # Get X most similar rows (nearest neighbors).
  # By default, return the distances of all rows. This will be quite inefficient, so override this when looking for a smaller subset by specifying a smaller number when invoking #kneighbors.
  # A formula of "minkowski" and p of 2 makes for a Euclidean distance metric.
  # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
  nearest_neighbors_model = NearestNeighbors(
    n_neighbors = len(data_frame),
    algorithm = 'auto',
    metric = 'minkowski',
    p = 2
  ).fit(data_frame)
  all_distances, all_indices = nearest_neighbors_model.kneighbors(random_row)
  nearest_neighbors_distances, nearest_neighbors_indices = nearest_neighbors_model.kneighbors(random_row, n_neighbors = 5)

  # Calculate a "similarity score" of each neighbor. This is not the absolute similarity of the neighbor from the target
  # one, it's more of its percentile ranking within the distances of all rows, meaning that it's effectively its ranking
  # within the population, converted to a percentage/score.
  nearest_neighbors_similarity_score = [100 - stats.percentileofscore(all_distances[0], distance, 'rank') for distance in nearest_neighbors_distances[0]]

  # Map the neighbors' result set indices to actual rows.
  nearest_neighbors = data_frame.iloc[nearest_neighbors_indices[0]]

  print('Random row:\n', random_row, '\n\n')
  print('All distances (closer to zero the better):\n', all_distances[0], '\n\n')
  print('Nearest Neighbors indices (within the result set, not initial data set):\n', nearest_neighbors_indices[0], '\n\n')
  print('Nearest Neighbors distances (closer to zero the better):\n', nearest_neighbors_distances[0], '\n\n')
  print('Nearest Neighbors similarity score (within population) (higher the better):\n', nearest_neighbors_similarity_score, '\n\n')
  print('Nearest Neighbors:\n', nearest_neighbors)

if __name__ == '__main__':
  main()
