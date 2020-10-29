from matchmaker import *

def main():
  ARGUMENTS = Arguments.parse_arguments()

  # This command-line interface is more for testing than actual use, so hard-code test input values in.
  # The web interface allows for dynamic values to be passed in for "real" use.
  #
  # Note that the following fields are about the user: age, relationship_status, sex, sexual_orientation. All other
  # fields represent what attributes the user is looking for.
  input_data = [
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
  # input_data[] = ''

  input_data_frame, nearest_neighbors_data_frame = Model.execute(
    input_data = input_data,
    force_training = ARGUMENTS.force_training,
    matches_to_retrieve = ARGUMENTS.matches_to_retrieve
  )

  if len(nearest_neighbors_data_frame) == 0:
    print('No matches :(')
  else:
    print('Input data:\n', input_data_frame, '\n\n')
    print('Nearest Neighbors:\n', nearest_neighbors_data_frame)

if __name__ == '__main__':
  main()
