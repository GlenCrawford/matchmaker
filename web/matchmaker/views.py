from django.shortcuts import render
from django.http import HttpResponse
from .forms import ProfileForm
import matchmaker as Matchmaker

def home(request):
  return render(request, 'home.html')

def profile(request):
  context = {
    'profile_form': ProfileForm()
  }

  return render(request, 'profile.html', context)

def matches(request):
  profile_form = ProfileForm(request.POST)

  if profile_form.is_valid():
    input_data_frame, matches_data_frame = Matchmaker.Model.execute(
      input_data = [
        profile_form.cleaned_data['age'],
        profile_form.cleaned_data['relationship_status'],
        profile_form.cleaned_data['sex'],
        profile_form.cleaned_data['sexual_orientation'],
        profile_form.cleaned_data['body_type'],
        profile_form.cleaned_data['diet'],
        profile_form.cleaned_data['drinks'],
        profile_form.cleaned_data['drugs'],
        profile_form.cleaned_data['education'],
        profile_form.cleaned_data['ethnicity'],
        profile_form.cleaned_data['offspring'],
        profile_form.cleaned_data['pets'],
        profile_form.cleaned_data['religion'],
        profile_form.cleaned_data['smokes'],
        profile_form.cleaned_data['speaks']
      ],
      force_training = True,
      matches_to_retrieve = 50
    )

    if len(matches_data_frame) == 0:
      return HttpResponse('No matches :(')
    else:
      context = {
        matches: [Matchmaker.Match.Match(match_row) for index, match_row in matches_data_frame.iterrows()]
      }

      return render(request, 'matches.html', context)
  else:
    return HttpResponse(profile_form.errors.as_json(), status = 422, content_type = 'application/json')
