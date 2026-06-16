from django.contrib.auth.models import User
from rest_framework import serializers

from .models import LoanApplication, Decision, AltData


class RegisterSerializer(serializers.ModelSerializer):
    """User sign-up."""
    password = serializers.CharField(write_only=True, min_length=6)

    class Meta:
        model = User
        fields = ["id", "username", "email", "password"]

    def create(self, validated_data):
        return User.objects.create_user(**validated_data)


class DecisionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Decision
        fields = ["decision", "score", "probability", "threshold",
                  "model_version", "reasons", "summary", "explanation", "created_at"]


class AltDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = AltData
        fields = ["source", "features", "created_at"]


class LoanApplicationSerializer(serializers.ModelSerializer):
    """Validates the application input and exposes the model's decision."""
    decision = DecisionSerializer(read_only=True)
    altdata = AltDataSerializer(read_only=True)

    class Meta:
        model = LoanApplication
        fields = ["id", "income", "loan_amount", "property_value",
                  "loan_to_value_ratio", "dti", "loan_term", "loan_type",
                  "loan_purpose", "lien_status", "occupancy_type",
                  "created_at", "decision", "altdata"]
        read_only_fields = ["id", "created_at", "decision", "altdata"]
