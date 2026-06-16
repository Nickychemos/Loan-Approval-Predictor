from django.conf import settings
from django.db import models


class LoanApplication(models.Model):
    """A housing-backed loan application submitted by a user.

    Fields mirror the model's input features. property_value is the value of
    the mortgaged property (collateral); loan_to_value_ratio = loan / property.
    """
    LOAN_TYPE = [("1", "Conventional"), ("2", "FHA"), ("3", "VA"), ("4", "RHS/FSA")]
    LOAN_PURPOSE = [("1", "Purchase"), ("2", "Home improvement"),
                    ("31", "Refinancing"), ("32", "Cash-out refinancing"), ("4", "Other")]
    LIEN_STATUS = [("1", "First lien"), ("2", "Subordinate lien")]
    OCCUPANCY = [("1", "Principal residence"), ("2", "Second residence"), ("3", "Investment")]

    applicant = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                                  related_name="applications")

    # numeric features
    income = models.FloatField(help_text="Annual income ($)", null=True, blank=True)
    loan_amount = models.FloatField(help_text="Requested loan amount ($)")
    property_value = models.FloatField(help_text="Collateral/property value ($)", null=True, blank=True)
    loan_to_value_ratio = models.FloatField(help_text="loan / property value (%)", null=True, blank=True)
    dti = models.FloatField(help_text="Debt-to-income ratio (%)", null=True, blank=True)
    loan_term = models.FloatField(help_text="Term (months)", null=True, blank=True, default=360)

    # categorical features
    loan_type = models.CharField(max_length=2, choices=LOAN_TYPE, default="1")
    loan_purpose = models.CharField(max_length=2, choices=LOAN_PURPOSE, default="1")
    lien_status = models.CharField(max_length=2, choices=LIEN_STATUS, default="1")
    occupancy_type = models.CharField(max_length=2, choices=OCCUPANCY, default="1")

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"App #{self.pk} by {self.applicant} (${self.loan_amount:,.0f})"


class Decision(models.Model):
    """The model's decision for an application (immutable audit record)."""
    application = models.OneToOneField(LoanApplication, on_delete=models.CASCADE,
                                       related_name="decision")
    decision = models.CharField(max_length=10)              # "approve" / "deny"
    score = models.FloatField()                             # 0-100
    probability = models.FloatField()                       # calibrated P(approve)
    threshold = models.FloatField()
    model_version = models.CharField(max_length=120)
    reasons = models.JSONField(default=list)                # structured SHAP contributions
    summary = models.TextField(blank=True, default="")      # one-line plain-English statement
    explanation = models.JSONField(default=list)            # principal reasons (adverse-action for denials)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.decision.upper()} ({self.score}) for app #{self.application_id}"


class AuditLog(models.Model):
    """Append-only log of notable actions for compliance/traceability."""
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
                             null=True, blank=True)
    action = models.CharField(max_length=80)
    detail = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.created_at:%Y-%m-%d %H:%M} {self.action}"


class AltData(models.Model):
    """Alternative-data features extracted (via LlamaIndex/LLM) from a borrower's
    SMS or statements. Captured now; fed to the model once we retrain on data
    that includes these signals."""
    application = models.OneToOneField(LoanApplication, on_delete=models.CASCADE,
                                       related_name="altdata")
    source = models.CharField(max_length=10, default="sms")   # sms / pdf
    features = models.JSONField(default=dict)                  # extracted AltDataFeatures
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"AltData ({self.source}) for app #{self.application_id}"
