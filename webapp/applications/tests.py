"""End-to-end tests for the loan-approval API.

These guard the CORE behaviour: a strong applicant is approved, a weak applicant
is denied, decisions are persisted/audited, auth is enforced, and users only see
their own applications. They exercise the real trained model via the DRF view.
"""
from unittest.mock import patch

from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.test import APITestCase

from applications.models import LoanApplication, Decision, AuditLog, AltData

STRONG = {"income": 180000, "loan_amount": 200000, "property_value": 500000,
          "loan_to_value_ratio": 40, "dti": 18, "loan_term": 360,
          "loan_type": "1", "loan_purpose": "1", "lien_status": "1", "occupancy_type": "1"}
WEAK = {"income": 32000, "loan_amount": 280000, "property_value": 300000,
        "loan_to_value_ratio": 96, "dti": 61, "loan_term": 360,
        "loan_type": "1", "loan_purpose": "1", "lien_status": "1", "occupancy_type": "1"}


class LoanFlowTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user("alice", "a@e.com", "pw123456")
        self.client.force_authenticate(self.user)

    def _submit(self, body):
        r = self.client.post("/api/applications/", body, format="json")
        self.assertEqual(r.status_code, status.HTTP_201_CREATED, r.content)
        return r.json()

    def test_strong_applicant_is_approved(self):
        d = self._submit(STRONG)["decision"]
        self.assertEqual(d["decision"], "approve")
        self.assertGreater(d["score"], 80)
        self.assertTrue(d["reasons"])  # SHAP reasons present

    def test_weak_applicant_is_denied(self):
        d = self._submit(WEAK)["decision"]
        self.assertEqual(d["decision"], "deny")
        self.assertLess(d["score"], 40)

    def test_denial_has_adverse_action_reasons(self):
        d = self._submit(WEAK)["decision"]
        self.assertIn("declined", d["summary"].lower())
        self.assertTrue(d["explanation"])                      # principal reasons present
        self.assertTrue(all(isinstance(x, str) for x in d["explanation"]))

    def test_approval_has_supporting_factors(self):
        d = self._submit(STRONG)["decision"]
        self.assertIn("approved", d["summary"].lower())
        self.assertTrue(d["explanation"])

    def test_strong_scores_higher_than_weak(self):
        s = self._submit(STRONG)["decision"]["score"]
        # fresh applicant so counts are clean across the OneToOne decision
        User.objects.create_user("carol", "c@e.com", "pw123456")
        self.client.force_authenticate(User.objects.get(username="carol"))
        w = self._submit(WEAK)["decision"]["score"]
        self.assertGreater(s, w)

    def test_decision_is_persisted_and_audited(self):
        self._submit(STRONG)
        self.assertEqual(LoanApplication.objects.count(), 1)
        self.assertEqual(Decision.objects.count(), 1)
        self.assertEqual(AuditLog.objects.filter(action="loan_decision").count(), 1)
        dec = Decision.objects.first()
        self.assertIn(dec.decision, ("approve", "deny"))
        self.assertTrue(dec.model_version.endswith(".joblib"))

    def test_registration_is_open(self):
        self.client.force_authenticate(None)
        r = self.client.post("/api/register/",
                             {"username": "bob", "password": "pw123456"}, format="json")
        self.assertEqual(r.status_code, status.HTTP_201_CREATED)

    def test_anonymous_cannot_apply(self):
        self.client.force_authenticate(None)
        r = self.client.post("/api/applications/", STRONG, format="json")
        self.assertIn(r.status_code, (status.HTTP_401_UNAUTHORIZED,
                                      status.HTTP_403_FORBIDDEN))

    @patch("loan_predictor.altdata.extract_features")
    def test_altdata_extracted_and_stored(self, mock_extract):
        """Posting text to /altdata/ extracts features (LLM mocked) and stores them."""
        from loan_predictor.altdata import AltDataFeatures
        mock_extract.return_value = AltDataFeatures(
            active_loans=2, lenders=["Tala", "Branch"], has_regular_salary=True)
        app_id = self._submit(STRONG)["id"]
        r = self.client.post(f"/api/applications/{app_id}/altdata/",
                             {"text": "Tala loan ... Branch repayment ..."}, format="json")
        self.assertEqual(r.status_code, status.HTTP_201_CREATED, r.content)
        self.assertEqual(r.json()["features"]["active_loans"], 2)
        self.assertEqual(AltData.objects.count(), 1)
        self.assertEqual(AuditLog.objects.filter(action="altdata_extracted").count(), 1)

    def test_users_only_see_their_own_applications(self):
        self._submit(STRONG)
        eve = User.objects.create_user("eve", "e@e.com", "pw123456")
        self.client.force_authenticate(eve)
        r = self.client.get("/api/applications/")
        self.assertEqual(len(r.json()), 0)

    def test_jwt_login_returns_token_and_grants_access(self):
        """The SPA flow: register -> obtain JWT -> call protected endpoint."""
        self.client.force_authenticate(None)
        self.client.post("/api/register/",
                         {"username": "dan", "password": "pw123456"}, format="json")
        r = self.client.post("/api/token/",
                            {"username": "dan", "password": "pw123456"}, format="json")
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.assertIn("access", r.json())
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {r.json()['access']}")
        me = self.client.get("/api/me/")
        self.assertEqual(me.status_code, status.HTTP_200_OK)
        self.assertEqual(me.json()["username"], "dan")
