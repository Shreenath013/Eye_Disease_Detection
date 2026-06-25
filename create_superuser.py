import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Eye_Disease.settings")
django.setup()

from django.contrib.auth import get_user_model

User = get_user_model()

username = os.getenv("DJANGO_SUPERUSER_USERNAME")
email = os.getenv("DJANGO_SUPERUSER_EMAIL")
password = os.getenv("DJANGO_SUPERUSER_PASSWORD")

if username and email and password:
    if not User.objects.filter(username=username).exists():
        User.objects.create_superuser(
            username=username,
            email=email,
            password=password,
            is_staff=True,
            is_superuser=True,
        )
        print("✅ Superuser created successfully.")
    else:
        print("ℹ️ Superuser already exists.")
else:
    print("⚠️ Superuser environment variables not set.")