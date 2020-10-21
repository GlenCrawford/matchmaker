from django.shortcuts import render
import matchmaker as Matchmaker

def recommendations(request):
  recommendations = Matchmaker.Utilities.django_app_test
  context = {
    'recommendations': recommendations
  }
  return render(request, 'templates/matchmaker/recommendations.html', context)
