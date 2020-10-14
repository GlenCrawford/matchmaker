from matchmaker import *
from sklearn.neighbors import NearestNeighbors

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
  # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
  nearest_neighbors_model = NearestNeighbors(n_neighbors = 5, algorithm = 'auto').fit(data_frame)
  nearest_neighbors_distances, nearest_neighbors_indices = nearest_neighbors_model.kneighbors(random_row)

  # Map the neighbors' result set indices to actual rows.
  nearest_neighbors = data_frame.iloc[nearest_neighbors_indices[0]]

  print('Random row:\n', random_row, '\n\n')
  print('Nearest Neighbors indices (within the result set, not initial data set):\n', nearest_neighbors_indices[0], '\n\n')
  print('Nearest Neighbors distances (closer to zero the better):\n', nearest_neighbors_distances[0], '\n\n')
  print('Nearest Neighbors:\n', nearest_neighbors)

if __name__ == '__main__':
  main()
