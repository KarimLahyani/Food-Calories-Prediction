from django.urls import path
from . import views
 
urlpatterns = [
    path('', views.index, name='index'),
    path('api/predict/', views.predict_calories, name='predict_calories'),
    path('api/predictions/', views.get_predictions, name='get_predictions'),
] 