from matchmaker import *

def main():
  data_frame = DataPreprocessing.load_input_data()
  data_frame = DataPreprocessing.preprocess_input_data(data_frame)

  # print(data_frame['body_type'].describe())
  # print(data_frame['body_type'].value_counts(dropna = False))
  print(data_frame)

if __name__ == '__main__':
  main()
