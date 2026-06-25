from django.contrib import admin
from django.urls import path
from application import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.index, name ='home'),
    path('login/', views.login_view, name='login'),
    path('doctor_dashboard/', views.doctor_dashboard, name ='doctor_dashboard'),
    path('patient_dashboard/', views.patient_dashboard, name ='patient_dashboard'),
    path('register/', views.register, name ='register'),
    path('result/<int:diagnosis_id>/', views.result, name ='result'),
    path('upload/', views.upload, name ='upload'),
    path('logout/', views.user_logout, name='logout'),
    path('download/<int:diagnosis_id>/', views.download_pdf, name='download_pdf'),
]

if settings.DEBUG:

    urlpatterns += static(

        settings.MEDIA_URL,

        document_root=settings.MEDIA_ROOT

    )
