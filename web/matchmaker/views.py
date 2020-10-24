from django.shortcuts import render
from .forms import ProfileForm
import matchmaker as Matchmaker

def home(request):
  context = {}
  return render(request, 'home.html', context)

def profile(request):
  context = {
    'profile_form': ProfileForm()
  }
  return render(request, 'profile.html', context)

def matches(request):
  matches = 'Matchmaker matches (data)'
  profile_form = ProfileForm(request.POST)
  # if form.is_valid():
  context = {
    'matches': matches
  }
  return render(request, 'recommendations.html', context)
