from django.urls import path

from . import views

urlpatterns = [
  path('recommendations', views.recommendations)
]
