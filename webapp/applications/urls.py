from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from .views import ApplicationViewSet, RegisterView, MeView

router = DefaultRouter()
router.register("applications", ApplicationViewSet, basename="application")

urlpatterns = [
    # auth (for the React SPA)
    path("register/", RegisterView.as_view(), name="register"),
    path("token/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("me/", MeView.as_view(), name="me"),
    # resources
    path("", include(router.urls)),
]
