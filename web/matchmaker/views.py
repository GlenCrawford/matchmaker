from django.shortcuts import render
import matchmaker as Matchmaker

def recommendations(request):
  recommendations = 'Matchmaker recommendations (data)'
  context = {
    'recommendations': recommendations
  }
  return render(request, 'templates/matchmaker/recommendations.html', context)
