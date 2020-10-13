from matchmaker import *

from sklearn.neighbors import NearestNeighbors

def main():
  data_frame = DataPreprocessing.load_input_data()
  data_frame = DataPreprocessing.preprocess_input_data(data_frame)

  # print(data_frame['drugs'].describe())
  # print(data_frame['drugs'].value_counts(dropna = False))
  # print(data_frame)

  # Quick proof-of-concept testing of scikit-learn KNN.
  # data_frame.drop(columns=['relationship_status', 'sex', 'sexual_orientation'], inplace = True)
  data_frame = data_frame[['age']]
  random_row = data_frame.sample()
  index_of_random_row = random_row.iloc[[0]].index.item()
  data_frame.drop(index_of_random_row, inplace = True)
  print('Data frame:\n', data_frame, '\n\n')
  print('Random row:\n', random_row, '\n\n')

  # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors
  nearest_neighbors_model = NearestNeighbors(n_neighbors = 5).fit(data_frame)
  nearest_neighbors_distances, nearest_neighbors_indices = nearest_neighbors_model.kneighbors(random_row)

  print('Nearest Neighbors indices:\n', nearest_neighbors_indices[0], '\n\n')
  print('Nearest Neighbors distances:\n', nearest_neighbors_distances[0], '\n\n')
  print('Nearest Neighbors:\n\n', data_frame.loc[nearest_neighbors_indices[0]])

if __name__ == '__main__':
  main()
