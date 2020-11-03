import unittest
import pandas as pd

from matchmaker import Match

class TestMatch(unittest.TestCase):
  def setUp(self):
    match_data = pd.Series({
      'score': 75,
      'age': 28,
      'sex': 'f',
      'sexual_orientation': 'straight',
      'body_type': 'curvy',
      'diet': 'vegan',
      'drinks': 'often',
      'drugs': 'never',
      'education': 'less_than_high_school',
      'ethnicity': 'asian',
      'have_children': False,
      'want_children': True,
      'pets_cats': False,
      'pets_dogs': True,
      'religion': 'judaism',
      'smokes': 'sometimes',
      'speaks': 'russian'
    })

    self.match = Match.Match(match_data)

  def test_name(self):
    self.assertIsInstance(self.match.name(), str)

  def test_score(self):
    self.assertEqual(self.match.score(), '75')

  def test_age(self):
    self.assertEqual(self.match.age(), '28')

  def test_sex(self):
    self.assertEqual(self.match.sex(), 'Female')

  def test_sexual_orientation(self):
    self.assertEqual(self.match.sexual_orientation(), 'straight')

  def test_body_type(self):
    self.assertEqual(self.match.body_type(), 'Curvy')

  def test_diet(self):
    self.assertEqual(self.match.diet(), 'Vegan')

  def test_drinks(self):
    self.assertEqual(self.match.drinks(), 'Often')

  def test_drugs(self):
    self.assertEqual(self.match.drugs(), 'Never')

  def test_education(self):
    self.assertEqual(self.match.education(), 'Less than high school')

  def test_ethnicity(self):
    self.assertEqual(self.match.ethnicity(), 'Asian')

  def test_have_children(self):
    self.assertEqual(self.match.have_children(), 'Does not have children')

  def test_want_children(self):
    self.assertEqual(self.match.want_children(), 'wants them')

  def test_pets_cats(self):
    self.assertEqual(self.match.pets_cats(), 'Does not have cat(s)')

  def test_pets_dogs(self):
    self.assertEqual(self.match.pets_dogs(), 'has dog(s)')

  def test_religion(self):
    self.assertEqual(self.match.religion(), 'Judaism')

  def test_smokes(self):
    self.assertEqual(self.match.smokes(), 'Sometimes')

  def test_speaks(self):
    self.assertEqual(self.match.speaks(), 'Russian')

if __name__ == '__main__':
  unittest.main()
