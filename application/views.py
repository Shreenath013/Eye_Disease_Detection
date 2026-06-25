# ==============================
# IMPORTS
# ==============================

from django.core.paginator import Paginator
from PIL import Image as PilImage
import os
import logging
import json

from functools import wraps

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login as auth_login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import Q, Count
from django.db.models.functions import TruncMonth
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from calendar import month_abbr

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .ai_utils import generate_clinical_recommendation
from .models import FundusImage, MyopiaDiagnosis, Profile
from xml.sax.saxutils import escape

logger = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================

def role_required(role):
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return redirect("login")
            try:
                profile = request.user.profile
            except Profile.DoesNotExist:
                return redirect("home")
            if profile.role != role:
                return redirect("home")
            return view_func(
                request,
                *args,
                **kwargs
            )
        return wrapper
    return decorator


def get_user_role(user):
   
    try:
        return user.profile.role
    except Profile.DoesNotExist:
        return None


def get_patient_name(diagnosis):
    
    try:
        return diagnosis.fundus_image.patient.profile.full_name
    except Exception:
        return diagnosis.fundus_image.patient.username


def calculate_severity(predicted_label, confidence):

    if predicted_label == "normal":
        return "None"
    elif predicted_label == "myopia":
        if confidence < 70:
            return "Mild"
        elif confidence < 90:
            return "Moderate"
        else:
            return "Severe"
    elif predicted_label == "pathological_myopia":
        if confidence < 80:
            return "Moderate"
        else:
            return "Severe"
    else:
        return "Unknown"


# =============================================================================
# INDEX
# =============================================================================

def index(request):
    """Public home page. Shows aggregate stats."""

    total_users = User.objects.filter(
        is_superuser=False
    ).count()
    total_images = FundusImage.objects.count()
    total_diagnosis = MyopiaDiagnosis.objects.count()

    context = {
        "total_users": total_users,
        "total_images": total_images,
        "total_diagnosis": total_diagnosis,
    }

    return render(request, "index.html", context)


# =============================================================================
# AUTHENTICATION
# =============================================================================

def login_view(request):

    if request.user.is_authenticated:
        role = get_user_role(request.user)
        if role == "doctor":
            return redirect("doctor_dashboard")
        return redirect("patient_dashboard")

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")

        if not username or not password:
            messages.error(request, "Please enter both username and password.")
            return render(request, "login.html")

        user = authenticate(request, username=username, password=password)

        if user is not None:
            try:
                profile = Profile.objects.get(user=user)

                if profile.role == "doctor" and not profile.is_verified:
                    messages.error(
                        request,
                        "Your doctor account is pending administrator verification. "
                        "Please contact the system administrator."
                    )
                    return render(request, "login.html")

                auth_login(request, user)
                
                remember = request.POST.get("remember")
                if not remember:
                    request.session.set_expiry(0)

                if profile.role == "doctor":
                    return redirect("doctor_dashboard")
                else:
                    return redirect("patient_dashboard")

            except Profile.DoesNotExist:
                messages.error(
                    request,
                    "Account profile not found. Please contact the administrator."
                )
                return render(request, "login.html")

        else:
            messages.error(request, "Invalid username or password.")

    return render(request, "login.html")


def user_logout(request):

    logout(request)
    return redirect("home")


def register(request):
    if request.user.is_authenticated:
        return redirect("home")

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        email = request.POST.get("email", "").strip().lower()
        full_name = request.POST.get("full_name", "").strip()
        role = request.POST.get("role", "").strip()
        password = request.POST.get("password", "")
        confirm_password = request.POST.get("confirm_password", "")
        
        if not all([username, email, full_name, role, password, confirm_password]):
            messages.error(request, "All fields are required.")
            return render(
                request,
                "register.html",
                {
                    "form_data": request.POST
                }
            )

        if role not in ["doctor", "patient"]:
            messages.error(request, "Invalid role selected.")
            return render(
                request,
                "register.html",
                {
                    "form_data": request.POST
                }
            )

        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return render(
                request,
                "register.html",
                {
                    "form_data": request.POST
                }
            )

        try:
            validate_password(password)
        except ValidationError as e:
            for error in e.messages:
                messages.error(request, error)
            return render(
                request,
                "register.html",
                {
                    "form_data": request.POST
                }
            )

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken.")
            return render(
                request,
                "register.html",
                {
                    "form_data": request.POST
                }
            )

        if User.objects.filter(email=email).exists():
            messages.error(request, "Account could not be created with these details.")
            return render(
                request,
                "register.html",
                {
                    "form_data": request.POST
                }
            )

        try:
            with transaction.atomic():
                user = User.objects.create_user(
                    username=username,
                    email=email,
                    password=password,
                )
                Profile.objects.create(
                    user=user,
                    role=role,
                    full_name=full_name,
                    email=email,
                    is_verified=(role == "patient"),
                )
        except Exception as e:
            logger.error("Registration failed: %s", e)
            messages.error(request, "Registration failed due to a server error. Please try again.")
            return render(
                request,
                "register.html",
                {
                    "form_data": request.POST
                }
            )

        if role == "doctor":
            messages.success(
                request,
                "Doctor account created. Your account is pending administrator verification before you can log in."
            )
        else:
            messages.success(request, "Account created successfully. You can now log in.")

        return redirect("login")

    return render(request, "register.html")


# =============================================================================
# UPLOAD & ANALYSIS
# =============================================================================

@login_required
@role_required("patient")
def upload(request):

    if request.method == "POST":

        image = request.FILES.get("image")

        if not image:
            messages.error(request, "No image file was uploaded.")
            return redirect("upload")

        allowed_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

        file_ext = os.path.splitext(
            image.name
        )[1].lower()

        if file_ext not in allowed_extensions:

            messages.error(
                request,
                "Invalid file type. Please upload a JPG, PNG, or TIFF image."
            )

            return redirect("upload")

        if image.size > 10 * 1024 * 1024:

            messages.error(
                request,
                "File too large. Maximum allowed size is 10 MB."
            )

            return redirect("upload")

        # Verify uploaded file is a real image

        try:

            img = PilImage.open(image)

            # Prevent extremely large images
            if img.width > 5000 or img.height > 5000:

                messages.error(
                    request,
                    "Image dimensions are too large."
                )

                return redirect("upload")

            # Reset file pointer after Pillow validation
            image.seek(0)

        except Exception as e:

            logger.error(
                "Invalid uploaded image: %s",
                e
            )

            messages.error(
                request,
                "Uploaded file is not a valid image."
            )

            return redirect("upload")

        notes = request.POST.get(
            "clinical_notes",
            ""
        ).strip() or None

        # =========================
        # INITIALIZE FOR CLEANUP
        # =========================

        fundus = None

        try:

            # =========================
            # ENTIRE PIPELINE IN ONE TRANSACTION
            # =========================

            with transaction.atomic():

                # =========================
                # CREATE FUNDUS RECORD
                # =========================

                fundus = FundusImage.objects.create(
                    patient=request.user,
                    image=image,
                    clinical_notes=notes,
                )

                image_path = fundus.image.path

                # =========================
                # ML PIPELINE
                # =========================
                
                from .ml_model.efficientnet_gradcam import generate_gradcam
                from .ml_model.efficientnet_predict import predict_image

                predicted_label, confidence, probabilities = (
                    predict_image(image_path)
                )

                severity = calculate_severity(
                    predicted_label,
                    confidence
                )

                gradcam_path = generate_gradcam(
                    image_path
                )

                relative_path = os.path.relpath(
                    gradcam_path,
                    settings.MEDIA_ROOT
                )

                ai_recommendation = (
                    generate_clinical_recommendation(
                        diagnosis=predicted_label,
                        confidence=confidence,
                        clinical_notes=notes or "",
                    )
                )

                # =========================
                # SAVE DIAGNOSIS
                # =========================

                diagnosis = MyopiaDiagnosis.objects.create(
                    fundus_image=fundus,
                    diagnosis=predicted_label,
                    confidence_score=confidence,
                    myopia_probability=probabilities["myopia"],
                    normal_probability=probabilities["normal"],
                    pathological_probability=probabilities["pathological_myopia"],
                    severity_level=severity,
                    recommendation=ai_recommendation,
                    segmentation_mask=relative_path,
                )

        except Exception as e:

            logger.error(
                "Upload pipeline failed for user %s: %s",
                request.user.username,
                e
            )

            # =========================
            # CLEANUP ORPHAN FILES
            # =========================

            if fundus:

                try:

                    if (
                        fundus.image
                        and os.path.exists(fundus.image.path)
                    ):

                        os.remove(fundus.image.path)

                except Exception as cleanup_error:

                    logger.error(
                        "Cleanup failed after upload pipeline error: %s",
                        cleanup_error
                    )

            messages.error(

                request,

                "Analysis failed due to a server error. "
                "Please try again. If the problem persists, "
                "contact the administrator."

            )

            return redirect("upload")

        return redirect(
            "result",
            diagnosis_id=diagnosis.id
        )

    return render(request, "upload.html")


# =============================================================================
# RESULT VIEW
# =============================================================================

@login_required
def result(request, diagnosis_id):

    diagnosis = get_object_or_404(MyopiaDiagnosis, id=diagnosis_id)

    user_role = get_user_role(request.user)

    if user_role == "patient":
        if diagnosis.fundus_image.patient != request.user:
            messages.error(request, "You are not authorized to view this report.")
            return redirect("patient_dashboard")

    elif user_role == "doctor":
        if diagnosis.assigned_doctor is None:
            diagnosis.assigned_doctor = request.user
            diagnosis.save(update_fields=["assigned_doctor"])

        elif diagnosis.assigned_doctor != request.user:
            messages.error(
                request,
                "This case is already assigned to another doctor."
            )
            return redirect("doctor_dashboard")

    else:

        messages.error(
            request,
            "Your account profile is missing. Contact the administrator."
        )

        return redirect("home")

    if request.method == "POST" and user_role == "doctor":

        if diagnosis.reviewed_by and diagnosis.reviewed_by != request.user:
            messages.error(request, "This case has already been reviewed by another doctor.")
            return redirect("doctor_dashboard")

        comment = request.POST.get("doctor_comment", "").strip()

        diagnosis.doctor_comment = comment or None
        diagnosis.is_reviewed = True
        diagnosis.reviewed_by = request.user
        diagnosis.reviewed_at = timezone.now()
        diagnosis.save(

            update_fields=[

                "doctor_comment",

                "is_reviewed",

                "reviewed_by",

                "reviewed_at"

            ]

        )

        diagnosis.refresh_from_db()

        messages.success(request, "Clinical comment saved successfully.")
        return redirect("result", diagnosis_id=diagnosis.id)
        
    reviewed_by_name = None

    if diagnosis.reviewed_by:

        try:

            reviewed_by_name = (

                diagnosis.reviewed_by.profile.full_name

                or diagnosis.reviewed_by.username

            )

        except Profile.DoesNotExist:

            reviewed_by_name = (

                diagnosis.reviewed_by.username

            )

    context = {
        "diagnosis": diagnosis,
        "user_role": user_role,
        "reviewed_by_name": reviewed_by_name,
    }

    return render(
        request,
        "result.html",
        context
    )


# =============================================================================
# PATIENT DASHBOARD
# =============================================================================

@login_required
@role_required("patient")
def patient_dashboard(request):
    
    diagnoses = MyopiaDiagnosis.objects.filter(
        fundus_image__patient=request.user
    ).select_related(
        "fundus_image",
        "reviewed_by",
    ).order_by("-created_at")

    total_reports = diagnoses.count()
    completed     = diagnoses.filter(is_reviewed=True).count()
    pending       = diagnoses.filter(is_reviewed=False).count()
    flagged       = diagnoses.filter(diagnosis="pathological_myopia").count()

    latest = diagnoses.first()
    
    # Paginate diagnosis history

    paginator = Paginator(

        diagnoses,

        10  # diagnoses per page

    )

    page_number = request.GET.get("page")

    page_obj = paginator.get_page(page_number)

    context = {
        "diagnoses": page_obj,
        "total_reports": total_reports,
        "completed": completed,
        "pending": pending,
        "flagged": flagged,
        "latest": latest,
    }

    return render(request, "patient_dashboard.html", context)


# =============================================================================
# DOCTOR DASHBOARD
# =============================================================================

@login_required
@role_required("doctor")
def doctor_dashboard(request):

    # ==============================
    # BASE QUERYSET
    # ==============================

    base_queryset = MyopiaDiagnosis.objects.select_related(
        "fundus_image",
        "fundus_image__patient",
        "fundus_image__patient__profile",
        "assigned_doctor",
        "assigned_doctor__profile",
        "reviewed_by",
        "reviewed_by__profile",
    ).filter(
        Q(assigned_doctor=request.user) |
        Q(assigned_doctor__isnull=True)
    ).order_by("-created_at")

    # ==============================
    # GLOBAL DASHBOARD STATS
    # ==============================

    total_cases = base_queryset.count()

    total_patients = User.objects.filter(
        profile__role="patient",
        is_staff=False,
        is_superuser=False,
    ).count()

    pending_review = base_queryset.filter(
        is_reviewed=False
    ).count()

    critical_alerts = base_queryset.filter(
        diagnosis="pathological_myopia"
    ).count()

    normal_count = base_queryset.filter(
        diagnosis="normal"
    ).count()

    myopia_count = base_queryset.filter(
        diagnosis="myopia"
    ).count()

    pathological_count = base_queryset.filter(
        diagnosis="pathological_myopia"
    ).count()

    mild_count = base_queryset.filter(
        severity_level="Mild"
    ).count()

    moderate_count = base_queryset.filter(
        severity_level="Moderate"
    ).count()

    severe_count = base_queryset.filter(
        severity_level="Severe"
    ).count()

    reviewed_count = base_queryset.filter(
        is_reviewed=True
    ).count()

    unreviewed_count = base_queryset.filter(
        is_reviewed=False
    ).count()

    # ==============================
    # SEARCH FILTER FOR TABLE ONLY
    # ==============================

    diagnosis_list = base_queryset

    search_query = request.GET.get(
        "search",
        ""
    ).strip()

    if search_query:

        diagnosis_list = diagnosis_list.filter(

            Q(
                fundus_image__patient__profile__full_name__icontains=search_query
            ) |

            Q(
                fundus_image__patient__username__icontains=search_query
            ) |

            Q(
                diagnosis__icontains=search_query
            )

        )

    # ==============================
    # RECENT TABLE RESULTS
    # ==============================

    recent_diagnoses = diagnosis_list[:10]

    # ==============================
    # MONTHLY CASE VOLUME
    # ==============================

    monthly_data = (
        base_queryset
        .annotate(month=TruncMonth("created_at"))
        .values("month")
        .annotate(total=Count("id"))
        .order_by("month")
    )

    # Dictionary: {3:3, 5:1, 6:3}
    monthly_dict = {
        item["month"].month: item["total"]
        for item in monthly_data
    }

    monthly_labels = [
        month_abbr[i]
        for i in range(1, 13)
    ]

    monthly_counts = [
        monthly_dict.get(i, 0)
        for i in range(1, 13)
    ]

    # ==============================
    # RECENT PATIENTS
    # ==============================

    recent_patients = User.objects.filter(
        profile__role="patient",
        is_staff=False,
        is_superuser=False,
    ).select_related(
        "profile"
    ).order_by("-date_joined")[:10]

    context = {
        "diagnoses": recent_diagnoses,
        "recent_patients": recent_patients,
        "total_cases": total_cases,
        "total_patients": total_patients,
        "pending_review": pending_review,
        "critical_alerts": critical_alerts,
        "normal_count": normal_count,
        "myopia_count": myopia_count,
        "pathological_count": pathological_count,
        "mild_count": mild_count,
        "moderate_count": moderate_count,
        "severe_count": severe_count,
        "reviewed_count": reviewed_count,
        "unreviewed_count": unreviewed_count,
        "search_query": search_query,
        "monthly_labels": json.dumps(monthly_labels),
        "monthly_counts": json.dumps(monthly_counts),

        # table results count
        "total_shown": min(10, diagnosis_list.count()),
    }

    return render(
        request,
        "doctor_dashboard.html",
        context
    )


# =============================================================================
# DOWNLOAD PDF REPORT
# =============================================================================

@login_required
def download_pdf(request, diagnosis_id):

    diagnosis = get_object_or_404(MyopiaDiagnosis, id=diagnosis_id)

    user_role = get_user_role(request.user)

    if user_role == "patient":
        if diagnosis.fundus_image.patient != request.user:
            messages.error(request, "You are not authorized to download this report.")
            return redirect("patient_dashboard")

    elif user_role == "doctor":

        # Automatically assign the case
        # to the first doctor downloading the PDF

        if diagnosis.assigned_doctor is None:

            diagnosis.assigned_doctor = request.user

            diagnosis.save(
                update_fields=["assigned_doctor"]
            )

        elif diagnosis.assigned_doctor != request.user:

            messages.error(

                request,

                "You are not authorized "
                "to download this report."

            )

            return redirect("doctor_dashboard")

    else:

        messages.error(
            request,
            "Your account profile is missing. Contact the administrator."
        )

        return redirect("home")

    response = HttpResponse(content_type="application/pdf")
    response["Content-Disposition"] = (
        f'attachment; filename="RetinaAI_Report_{diagnosis.id}.pdf"'
    )

    doc = SimpleDocTemplate(
        response,
        pagesize=A4,
        rightMargin=inch * 0.75,
        leftMargin=inch * 0.75,
        topMargin=inch * 0.75,
        bottomMargin=inch * 0.75,
    )

    styles = getSampleStyleSheet()
    elements = []

    elements.append(
        Paragraph(
            f"RetinaAI — AI Retinal Analysis Report #{diagnosis.id}",
            styles["Heading1"],
        )
    )
    elements.append(Spacer(1, 16))

    patient_name = get_patient_name(diagnosis)

    table_data = [
        ["Patient Name",  patient_name],
        ["Diagnosis",     diagnosis.diagnosis.replace("_", " ").title()],
        ["Severity",      diagnosis.severity_level or "N/A"],   # FIX: None guard
        ["Confidence",    f"{diagnosis.confidence_score:.2f}%"],
        [
            "Analysis Date",
            timezone.localtime(diagnosis.created_at).strftime("%d-%m-%Y %I:%M %p")
        ],
        ["Review Status", "Reviewed" if diagnosis.is_reviewed else "Pending Review"],
    ]

    if diagnosis.is_reviewed and diagnosis.reviewed_by:
        try:
            reviewer_name = diagnosis.reviewed_by.profile.full_name
        except Exception:
            reviewer_name = diagnosis.reviewed_by.username
        table_data.append(["Reviewed By", reviewer_name])

    table = Table(table_data, colWidths=[180, 295])
    table.setStyle(
        TableStyle([
            ("BACKGROUND",    (0, 0), (0, -1), colors.lightgrey),
            ("TEXTCOLOR",     (0, 0), (-1, -1), colors.black),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME",      (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ])
    )
    elements.append(table)
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Uploaded Fundus Image", styles["Heading2"]))
    elements.append(Spacer(1, 8))
    try:
        fundus_img = Image(
            diagnosis.fundus_image.image.path,
            width=3 * inch,
            height=3 * inch,
        )
        elements.append(fundus_img)
    except Exception:
        elements.append(Paragraph("(Fundus image file not found)", styles["BodyText"]))
    elements.append(Spacer(1, 20))

    if diagnosis.segmentation_mask:
        elements.append(Paragraph("AI Attention Heatmap (GradCAM)", styles["Heading2"]))
        elements.append(Spacer(1, 8))
        try:
            heatmap_img = Image(
                diagnosis.segmentation_mask.path,
                width=3 * inch,
                height=3 * inch,
            )
            elements.append(heatmap_img)
        except Exception:
            elements.append(Paragraph("(Heatmap image file not found)", styles["BodyText"]))
        elements.append(Spacer(1, 20))

    elements.append(Paragraph("AI Probability Breakdown", styles["Heading2"]))
    elements.append(Spacer(1, 8))

    prob_data = [
        ["Class",                "Probability"],
        ["Normal",               f"{diagnosis.normal_probability:.2f}%"],
        ["Myopia",               f"{diagnosis.myopia_probability:.2f}%"],
        ["Pathological Myopia",  f"{diagnosis.pathological_probability:.2f}%"],
    ]
    prob_table = Table(prob_data, colWidths=[250, 145])
    prob_table.setStyle(
        TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), colors.lightblue),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.black),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE",      (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("ALIGN",         (1, 0), (1, -1), "CENTER"),
        ])
    )
    elements.append(prob_table)
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("AI Clinical Recommendation", styles["Heading2"]))
    elements.append(Spacer(1, 8))
    elements.append(
        Paragraph(
            escape(
                diagnosis.recommendation or "Not available."
            ),
            styles["BodyText"]
        )
    )
    elements.append(Spacer(1, 20))

    if diagnosis.doctor_comment:
        elements.append(Paragraph("Doctor Review Comment", styles["Heading2"]))
        elements.append(Spacer(1, 8))
        elements.append(
            Paragraph(
                escape(diagnosis.doctor_comment or "None"),
                styles["BodyText"]
            )
        )
        elements.append(Spacer(1, 20))

    elements.append(Spacer(1, 10))
    elements.append(
        Paragraph(
            "<b>Disclaimer:</b> This report is generated by an AI model (EfficientNet-B3 + "
            "GradCAM). It constitutes decision support only and must be validated by a "
            "licensed ophthalmologist before any clinical decision is made.",
            styles["BodyText"],
        )
    )

    try:
        doc.build(elements)
    except Exception as e:
        logger.error("PDF generation failed for diagnosis %s: %s", diagnosis_id, e)
        return HttpResponse("PDF generation failed. Please try again.", status=500)

    return response
