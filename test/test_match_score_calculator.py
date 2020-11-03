import unittest
import pandas as pd

from matchmaker import MatchScoreCalculator
from matchmaker import DataPreprocessing

class TestMatchScoreCalculator(unittest.TestCase):
  def setUp(self):
    input_data_frame = pd.DataFrame(
      [
        [
          '30', # age
          'single', # relationship_status
          'm', # sex
          'straight', # sexual_orientation
          'thin', # body_type
          'anything', # diet
          'rarely', # drinks
          'never', # drugs
          'completed_undergraduate_study', # education
          'white', # ethnicity
          'no_kids', # offspring
          '', # pets
          'christianity', # religion
          'no', # smokes
          'english' # speaks
        ]
      ],
      columns = DataPreprocessing.INPUT_DATA_COLUMNS_TO_USE
    )

    matches_data_frame = pd.DataFrame(
      [
        # Identical to the input.
        [
          '30', # age
          'single', # relationship_status
          'f', # sex
          'straight', # sexual_orientation
          'thin', # body_type
          'anything', # diet
          'rarely', # drinks
          'never', # drugs
          'completed_undergraduate_study', # education
          'white', # ethnicity
          'no_kids', # offspring
          '', # pets
          'christianity', # religion
          'no', # smokes
          'english' # speaks
        ],

        # Multiple slight differences to the input (three features: age, body_type, pets).
        [
          '29', # age
          'single', # relationship_status
          'f', # sex
          'straight', # sexual_orientation
          'fit', # body_type
          'anything', # diet
          'rarely', # drinks
          'never', # drugs
          'completed_undergraduate_study', # education
          'white', # ethnicity
          'no_kids', # offspring
          'dogs', # pets
          'christianity', # religion
          'no', # smokes
          'english' # speaks
        ],

        # One significant difference to the input (offspring, a very highly-weighted feature).
        [
          '30', # age
          'single', # relationship_status
          'f', # sex
          'straight', # sexual_orientation
          'thin', # body_type
          'anything', # diet
          'rarely', # drinks
          'never', # drugs
          'completed_undergraduate_study', # education
          'white', # ethnicity
          'has_kids_but_no_more', # offspring
          '', # pets
          'christianity', # religion
          'no', # smokes
          'english' # speaks
        ],

        # Multiple moderate differences to the input (a lot of features: age, body_type, drinks, education, offspring, pets, religion).
        [
          '25', # age
          'single', # relationship_status
          'f', # sex
          'straight', # sexual_orientation
          'average', # body_type
          'anything', # diet
          'socially', # drinks
          'never', # drugs
          'completed_postgraduate_study', # education
          'white', # ethnicity
          'no_kids_dont_want_any', # offspring
          'cats,dogs', # pets
          'judaism', # religion
          'no', # smokes
          'english' # speaks
        ],

        # Multiple significant differences to the input (all used features).
        [
          '100', # age
          'single', # relationship_status
          'f', # sex
          'straight', # sexual_orientation
          'overweight', # body_type
          'vegan', # diet
          'often', # drinks
          'often', # drugs
          'less_than_high_school', # education
          'black', # ethnicity
          'has_kids_but_no_more', # offspring
          'cats,dogs', # pets
          'islam', # religion
          'yes', # smokes
          'english' # speaks
        ]
      ],
      columns = DataPreprocessing.INPUT_DATA_COLUMNS_TO_USE
    )

    input_data_frame = DataPreprocessing.preprocess_input_data(input_data_frame, use_fitted_encoders = True)
    matches_data_frame = DataPreprocessing.preprocess_input_data(matches_data_frame, use_fitted_encoders = True)

    self.match_scores = MatchScoreCalculator.calculate_match_score(input_data_frame, matches_data_frame)

  def test(self):
    self.assertEqual(self.match_scores, [100, 97, 77, 64, 10])
 
if __name__ == '__main__':
  unittest.main()
