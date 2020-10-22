from django.shortcuts import render
import matchmaker as Matchmaker

def home(request):
  context = {}
  return render(request, 'home.html', context)

def recommendations(request):
  recommendations = 'Matchmaker recommendations (data)'
  context = {
    'recommendations': recommendations
  }
  return render(request, 'recommendations.html', context)
