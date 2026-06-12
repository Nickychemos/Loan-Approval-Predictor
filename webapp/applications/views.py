from rest_framework import generics, viewsets, permissions
from rest_framework.response import Response
from rest_framework.views import APIView

from loan_predictor.predictor import predict

from .models import LoanApplication, Decision, AuditLog
from .serializers import RegisterSerializer, LoanApplicationSerializer

FEATURE_FIELDS = ["income", "loan_amount", "property_value", "loan_to_value_ratio",
                  "dti", "loan_term", "loan_type", "loan_purpose", "lien_status",
                  "occupancy_type"]


class RegisterView(generics.CreateAPIView):
    """POST /api/register/ — create a user account (open)."""
    serializer_class = RegisterSerializer
    permission_classes = [permissions.AllowAny]


class MeView(APIView):
    """GET /api/me/ — the currently authenticated user (for the SPA)."""
    def get(self, request):
        u = request.user
        return Response({"id": u.id, "username": u.username, "email": u.email})


class ApplicationViewSet(viewsets.ModelViewSet):
    """Submit and list housing loan applications. On submit, the model scores
    the application and the decision is stored against it."""
    serializer_class = LoanApplicationSerializer
    permission_classes = [permissions.IsAuthenticated]
    http_method_names = ["get", "post"]

    def get_queryset(self):
        return (LoanApplication.objects
                .filter(applicant=self.request.user)
                .select_related("decision")
                .order_by("-created_at"))

    def perform_create(self, serializer):
        app = serializer.save(applicant=self.request.user)
        features = {f: getattr(app, f) for f in FEATURE_FIELDS}
        result = predict(features)
        Decision.objects.create(
            application=app,
            decision=result["decision"],
            score=result["score"],
            probability=result["probability"],
            threshold=result["threshold"],
            model_version=result["model_version"],
            reasons=result["top_reasons"],
            summary=result["summary"],
            explanation=result["explanation"],
        )
        AuditLog.objects.create(
            user=self.request.user, action="loan_decision",
            detail={"application": app.pk, "decision": result["decision"],
                    "score": result["score"]},
        )
