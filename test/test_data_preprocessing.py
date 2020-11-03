import unittest
import pandas as pd
from pandas.testing import assert_frame_equal

from matchmaker import DataPreprocessing

class TestDataPreprocessing(unittest.TestCase):
  def setUp(self):
    raw_data_frame = pd.DataFrame(
      [
        [
          '30', # age
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
        ],
        [
          '25', # age
          'f', # sex
          'gay', # sexual_orientation
          'a little extra', # body_type
          'vegan', # diet
          'very often', # drinks
          'often', # drugs
          'dropped out of space camp', # education
          'hispanic / latin', # ethnicity
          'has kids, but doesn\'t want more', # offspring
          'has dogs and has cats', # pets
          'judaism and very serious about it', # religion
          'sometimes', # smokes
          'japanese, english' # speaks
        ],
        [
          '100', # age
          'f', # sex
          'gay', # sexual_orientation
          'overweight', # body_type
          'vegetarian', # diet
          'not at all', # drinks
          'never', # drugs
          'dropped out of med school', # education
          'black', # ethnicity
          'has kids, and wants more', # offspring
          'has dogs and dislikes cats', # pets
          'catholicism but not too serious about it', # religion
          'yes', # smokes
          'hindi (fluently), c++, russian' # speaks
        ]
      ],
      columns = [column for column in DataPreprocessing.INPUT_DATA_COLUMNS_TO_USE if column != 'relationship_status']
    )

    self.preprocessed_data_frame = DataPreprocessing.preprocess_input_data(raw_data_frame, use_fitted_encoders = True)

  def test(self):
    assert_frame_equal(
      self.preprocessed_data_frame, pd.DataFrame(
        columns = [
          'age',
          'sex',
          'sexual_orientation',
          'body_type',
          'diet',
          'drinks',
          'drugs',
          'education',
          'ethnicity_asian',
          'ethnicity_black',
          'ethnicity_hispanic_latin',
          'ethnicity_unknown',
          'ethnicity_white',
          'have_children',
          'want_children',
          'pets_cats',
          'pets_dogs',
          'religion_agnosticism',
          'religion_atheism',
          'religion_buddhism',
          'religion_catholicism',
          'religion_christianity',
          'religion_hinduism',
          'religion_islam',
          'religion_judaism',
          'religion_other',
          'religion_unknown',
          'smokes',
          'speaks'
        ],
        data = [
          [
            0.652174, # age
            'm', # sex
            'straight', # sexual_orientation
            0, # body_type
            1.5, # diet
            0.28, # drinks
            0, # drugs
            0.375, # education
            0, # ethnicity_asian
            0, # ethnicity_black
            0, # ethnicity_hispanic_latin
            0, # ethnicity_unknown
            0.1, # ethnicity_white
            0, # have_children
            1, # want_children
            0, # pets_cats
            0, # pets_dogs
            0, # religion_agnosticism
            0, # religion_atheism
            0, # religion_buddhism
            0, # religion_catholicism
            0.5, # religion_christianity
            0, # religion_hinduism
            0, # religion_islam
            0, # religion_judaism
            0, # religion_other
            0, # religion_unknown
            0, # smokes
            'english' # speaks
          ],
          [
            0.380434, # age
            'f', # sex
            'gay', # sexual_orientation
            0.24, # body_type
            0, # diet
            0.7, # drinks
            3, # drugs
            0.125, # education
            0, # ethnicity_asian
            0, # ethnicity_black
            0.1, # ethnicity_hispanic_latin
            0, # ethnicity_unknown
            0, # ethnicity_white
            1, # have_children
            0, # want_children
            0.1, # pets_cats
            0.1, # pets_dogs
            0, # religion_agnosticism
            0, # religion_atheism
            0, # religion_buddhism
            0, # religion_catholicism
            0, # religion_christianity
            0, # religion_hinduism
            0, # religion_islam
            0.5, # religion_judaism
            0, # religion_other
            0, # religion_unknown
            1.8, # smokes
            'japanese' # speaks
          ],
          [
            4.456521, # age
            'f', # sex
            'gay', # sexual_orientation
            0.4, # body_type
            0.375, # diet
            0, # drinks
            0, # drugs
            0.125, # education
            0, # ethnicity_asian
            0.1, # ethnicity_black
            0, # ethnicity_hispanic_latin
            0, # ethnicity_unknown
            0, # ethnicity_white
            1, # have_children
            1, # want_children
            0, # pets_cats
            0.1, # pets_dogs
            0, # religion_agnosticism
            0, # religion_atheism
            0, # religion_buddhism
            0.5, # religion_catholicism
            0, # religion_christianity
            0, # religion_hinduism
            0, # religion_islam
            0, # religion_judaism
            0, # religion_other
            0, # religion_unknown
            3.0, # smokes
            'hindi' # speaks
          ]
        ]
      ),
      check_dtype = False
    )

if __name__ == '__main__':
  unittest.main()
