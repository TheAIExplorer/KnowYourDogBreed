from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_view, name='upload'),
    path('', views.index, name='index'),      # URL pattern for the index view
    path('predict/', views.predict, name='predict'),  # URL pattern for the predict view
]
