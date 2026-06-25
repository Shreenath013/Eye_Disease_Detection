from django.contrib import admin
from .models import Profile, FundusImage, MyopiaDiagnosis


@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "role", "is_verified")
    list_filter = ("role", "is_verified")
    search_fields = ("user__username", "full_name", "email")


@admin.register(FundusImage)
class FundusImageAdmin(admin.ModelAdmin):
    list_display = ("id", "patient", "uploaded_at")
    list_filter = ("uploaded_at",)
    search_fields = ("patient__username",)


@admin.register(MyopiaDiagnosis)
class MyopiaDiagnosisAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "fundus_image",
        "diagnosis",
        "confidence_score",
        "created_at"
    )
    list_filter = ("diagnosis", "created_at")
    search_fields = ("fundus_image__patient__username",)