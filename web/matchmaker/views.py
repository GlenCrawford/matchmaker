from django.shortcuts import render
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
  context = {
    #
  }

  return render(request, 'matches.html', context)
