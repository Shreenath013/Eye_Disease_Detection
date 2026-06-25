from django.test import TestCase
from django.urls import reverse
from django.contrib.auth.models import User

from .models import Profile


class AuthenticationTests(TestCase):

    def setUp(self):
        self.patient = User.objects.create_user(
            username="patient1",
            password="Test@12345"
        )

        Profile.objects.create(
            user=self.patient,
            full_name="Test Patient",
            role="patient"
        )

        self.doctor = User.objects.create_user(
            username="doctor1",
            password="Test@12345"
        )

        Profile.objects.create(
            user=self.doctor,
            full_name="Test Doctor",
            role="doctor"
        )

    # -----------------------------
    # PUBLIC PAGES
    # -----------------------------

    def test_home_page_loads(self):
        response = self.client.get(reverse("home"))
        self.assertEqual(response.status_code, 200)

    def test_login_page_loads(self):
        response = self.client.get(reverse("login"))
        self.assertEqual(response.status_code, 200)

    def test_register_page_loads(self):
        response = self.client.get(reverse("register"))
        self.assertEqual(response.status_code, 200)

    # -----------------------------
    # LOGIN
    # -----------------------------

    def test_valid_patient_login(self):
        login = self.client.login(
            username="patient1",
            password="Test@12345"
        )

        self.assertTrue(login)

    def test_valid_doctor_login(self):
        login = self.client.login(
            username="doctor1",
            password="Test@12345"
        )

        self.assertTrue(login)

    def test_invalid_login(self):
        login = self.client.login(
            username="patient1",
            password="WrongPassword"
        )

        self.assertFalse(login)

    # -----------------------------
    # AUTHENTICATION
    # -----------------------------

    def test_patient_dashboard_requires_login(self):
        response = self.client.get(
            reverse("patient_dashboard")
        )

        self.assertEqual(response.status_code, 302)

    def test_doctor_dashboard_requires_login(self):
        response = self.client.get(
            reverse("doctor_dashboard")
        )

        self.assertEqual(response.status_code, 302)

    def test_upload_requires_login(self):
        response = self.client.get(
            reverse("upload")
        )

        self.assertEqual(response.status_code, 302)

    # -----------------------------
    # ROLE BASED ACCESS
    # -----------------------------

    def test_patient_cannot_access_doctor_dashboard(self):

        self.client.login(
            username="patient1",
            password="Test@12345"
        )

        response = self.client.get(
            reverse("doctor_dashboard")
        )

        self.assertEqual(response.status_code, 302)

    def test_doctor_cannot_access_patient_dashboard(self):

        self.client.login(
            username="doctor1",
            password="Test@12345"
        )

        response = self.client.get(
            reverse("patient_dashboard")
        )

        self.assertEqual(response.status_code, 302)

    def test_patient_can_access_upload_page(self):

        self.client.login(
            username="patient1",
            password="Test@12345"
        )

        response = self.client.get(
            reverse("upload")
        )

        self.assertEqual(response.status_code, 200)

    def test_doctor_can_access_dashboard(self):

        self.client.login(
            username="doctor1",
            password="Test@12345"
        )

        response = self.client.get(
            reverse("doctor_dashboard")
        )

        self.assertEqual(response.status_code, 200)