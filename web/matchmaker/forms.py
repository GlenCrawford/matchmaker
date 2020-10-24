from django import forms
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit

class ProfileForm(forms.Form):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.helper = FormHelper()
    self.helper.form_id = 'profile_form'
    self.helper.form_method = 'post'
    self.helper.form_action = 'matches'

    self.helper.add_input(Submit('submit', 'Submit'))

  # Fields.
  sex = forms.ChoiceField(widget = forms.RadioSelect, label = 'Sex', required = True, choices = [('m', 'Male'), ('f', 'Female')])
  sexual_orientation = forms.ChoiceField(widget = forms.RadioSelect, label = 'Sexual orientation', required = True, choices = [('straight', 'Straight'), ('gay', 'Gay'), ('bisexual', 'Bisexual')], initial = 'straight')
  age = forms.IntegerField(label = 'Age', required = True, min_value = 18, max_value = 110)
  relationship_status = forms.ChoiceField(label = 'Relationship status', required = True, disabled = True, choices = [('single', 'Single'), ('available', 'Available'), ('seeing someone', 'Seeing someone'), ('married', 'Married')], initial = 'single', help_text = 'Please don\'t select married...')
  ethnicity = forms.ChoiceField(label = 'Ethnicity', required = True, choices = [('', ''), ('white', 'White'), ('asian', 'Asian'), ('black', 'Black'), ('hispanic_latin', 'Hispanic / Latin')], initial = '')
  speaks = forms.ChoiceField(label = 'Language', required = True, choices = [('', ''), ('afrikaans', 'Afrikaans'), ('english', 'English'), ('french', 'French'), ('hindi', 'Hindi'), ('japanese', 'Japanese'), ('mandarin_chinese', 'Mandarin Chinese'), ('portuguese', 'Portuguese'), ('russian', 'Russian'), ('spanish', 'Spanish')], initial = '', help_text = 'Your most commonly spoken language.')
  religion = forms.ChoiceField(label = 'Religion', required = True, choices = [('', ''), ('atheism', 'Atheism'), ('agnosticism', 'Agnosticism'), ('buddhism', 'Buddhism'), ('hinduism', 'Hinduism'), ('islam', 'Islam'), ('judaism', 'Judaism'), ('christianity', 'Christianity'), ('catholicism', 'Catholicism'), ('other', 'Other')], initial = '')
  education = forms.ChoiceField(label = 'Education level', required = True, choices = [('', ''), ('less_than_high_school', 'Less than high school'), ('high_school', 'High school'), ('in_progress_study', 'Studying (in progress)'), ('completed_undergraduate_study', 'Completed undergraduate study'), ('completed_postgraduate_study', 'Completed postgraduate study')], initial = '')
  offspring = forms.ChoiceField(label = 'Children', required = True, choices = [('', ''), ('no_kids', 'None (but want them)'), ('no_kids_dont_want_any', 'None (and don\'t want any)'), ('has_kids', 'Have children (and want more)'), ('has_kids_but_no_more', 'Have children (but no more)')], initial = '')
  body_type = forms.ChoiceField(label = 'Body type', required = True, choices = [('', ''), ('thin', 'Thin'), ('fit', 'Fit'), ('average', 'Average'), ('curvy', 'Curvy'), ('overweight', 'Overweight')], initial = '')
  diet = forms.ChoiceField(label = 'Diet', required = True, choices = [('', ''), ('anything', 'Anything'), ('vegetarian', 'Vegetarian'), ('vegan', 'Vegan')], initial = '')
  drinks = forms.ChoiceField(label = 'Drinks?', required = True, choices = [('', ''), ('never', 'Never'), ('rarely', 'Rarely'), ('socially', 'Socially'), ('often', 'Often')], initial = '')
  smokes = forms.ChoiceField(label = 'Smokes?', required = True, choices = [('', ''), ('no', 'No'), ('yes', 'Yes'), ('sometimes', 'Sometimes')], initial = '')
  drugs = forms.ChoiceField(label = 'Drugs?', required = True, choices = [('', ''), ('never', 'Never'), ('sometimes', 'Sometimes'), ('often', 'Often')], initial = '')
  pets = forms.ChoiceField(label = 'Pets', required = False, choices = [('', ''), ('', 'None'), ('cats', 'Have cat(s)'), ('dogs', 'Have dog(s)'), ('cats,dogs', 'Have cat(s) and dog(s)')], initial = '', help_text = 'Only cats and/or dogs, unfortunately :(')
