import names

class Match:
  def __init__(self, match_row):
    self.__score = match_row['score']
    self.__age = match_row['age']
    self.__sex = match_row['sex']
    self.__sexual_orientation = match_row['sexual_orientation']
    self.__body_type = match_row['body_type']
    self.__diet = match_row['diet']
    self.__drinks = match_row['drinks']
    self.__drugs = match_row['drugs']
    self.__education = match_row['education']
    self.__ethnicity = match_row['ethnicity']
    self.__have_children = match_row['have_children']
    self.__want_children = match_row['want_children']
    self.__pets_cats = match_row['pets_cats']
    self.__pets_dogs = match_row['pets_dogs']
    self.__religion = match_row['religion']
    self.__smokes = match_row['smokes']
    self.__speaks = match_row['speaks']

  # The training data is anonymized, so generate random names to make the profile look more real.
  def name(self):
    name_gender = { 'm': 'male', 'f': 'female' }[self.__sex]
    return names.get_full_name(gender = name_gender)

  # The following are decorator methods, one per attribute.

  def score(self):
    return f'{self.__score:.2f}'

  def age(self):
    return f'{self.__age:.0f}'

  def sex(self):
    return { 'm': 'Male', 'f': 'Female' }[self.__sex]

  def sexual_orientation(self):
    return self.__sexual_orientation

  def body_type(self):
    return self.__body_type.capitalize()

  def diet(self):
    return self.__diet.capitalize()

  def drinks(self):
    return self.__drinks.capitalize()

  def drugs(self):
    return self.__drugs.capitalize()

  def education(self):
    return self.__education.capitalize().replace('_', ' ')

  def ethnicity(self):
    return self.__ethnicity.capitalize().replace('_', ' / ')

  def have_children(self):
    return 'Has children' if self.__have_children else 'Does not have children'

  def want_children(self):
    # TODO: Can make this better in combination with have_children.
    return 'wants them' if self.__want_children else 'don\'t want any/more'

  def pets_cats(self):
    return 'Has cat(s)' if self.__pets_cats else 'Does not have cat(s)'

  def pets_dogs(self):
    # TODO: Can maybe combine both pets and dogs into a better combined string.
    return 'has dog(s)' if self.__pets_dogs else 'does not have dog(s)'

  def religion(self):
    return self.__religion.capitalize()

  def smokes(self):
    return self.__smokes.capitalize()

  def speaks(self):
    return self.__speaks.capitalize()
