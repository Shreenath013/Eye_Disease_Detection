from django.db import models
from django.contrib.auth.models import User


# 1️⃣ Profile Model (Doctor / Patient Role)
class Profile(models.Model):

    ROLE_CHOICES = (
        ('doctor', 'Doctor'),
        ('patient', 'Patient'),
    )

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)

    full_name = models.CharField(max_length=150)
    email = models.EmailField()

    is_verified = models.BooleanField(default=False)  # For doctor approval

    def __str__(self):
        return f"{self.user.username} - {self.role}"


# 2️⃣ Fundus Image Upload Model
class FundusImage(models.Model):

    patient = models.ForeignKey(User, on_delete=models.CASCADE)

    image = models.ImageField(upload_to='fundus_images/')
    clinical_notes = models.TextField(blank=True, null=True)

    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Fundus Image {self.id} - {self.patient.username}"


# 3️⃣ Myopia Diagnosis Result Model
class MyopiaDiagnosis(models.Model):

    DIAGNOSIS_CHOICES = (
        ('normal', 'Normal'),
        ('myopia', 'Myopia'),
        ('pathological_myopia', 'Pathological Myopia'),
    )

    # 🔹 Link to uploaded fundus image
    fundus_image = models.OneToOneField(
        'FundusImage',
        on_delete=models.CASCADE
    )

    # 🔹 AI Prediction
    diagnosis = models.CharField(
        max_length=50,
        choices=DIAGNOSIS_CHOICES
    )

    confidence_score = models.FloatField()
    
    myopia_probability = models.FloatField(
        default=0
    )

    normal_probability = models.FloatField(
        default=0
    )

    pathological_probability = models.FloatField(
        default=0
    )

    severity_level = models.CharField(
        max_length=20,
        blank=True,
        null=True
    )

    recommendation = models.TextField()

    segmentation_mask = models.ImageField(
        upload_to='segmentation_masks/',
        blank=True,
        null=True
    )

    # 🔹 Doctor Review System (NEW)
    is_reviewed = models.BooleanField(default=False)

    reviewed_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="reviewed_cases"
    )
    
    assigned_doctor = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="assigned_cases"
    )

    reviewed_at = models.DateTimeField(
        null=True,
        blank=True
    )
    
    doctor_comment = models.TextField(
        blank=True,
        null=True
    )

    # 🔹 Timestamp
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Diagnosis for Image {self.fundus_image.id}"

    class Meta:
        verbose_name = "Myopia Diagnosis"
        verbose_name_plural = "Myopia Diagnoses"
