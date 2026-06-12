from django.contrib import admin

from .models import LoanApplication, Decision, AuditLog


@admin.register(LoanApplication)
class LoanApplicationAdmin(admin.ModelAdmin):
    list_display = ("id", "applicant", "loan_amount", "property_value",
                    "loan_to_value_ratio", "dti", "created_at")
    list_filter = ("loan_type", "loan_purpose", "occupancy_type", "created_at")
    search_fields = ("applicant__username",)


@admin.register(Decision)
class DecisionAdmin(admin.ModelAdmin):
    list_display = ("application", "decision", "score", "summary", "created_at")
    list_filter = ("decision", "created_at")


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ("created_at", "user", "action")
    list_filter = ("action", "created_at")
