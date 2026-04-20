"""
=============================================================
 Brain Tumor Detection — Automated Test Suite (pytest)
=============================================================
 Run with:  pytest test_app.py -v
 Requirements: pip install pytest pillow numpy
=============================================================
"""

import pytest
import io
import os
import json
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock

# ── Import your Flask app ──────────────────────────────────
# Make sure main.py is in the same folder, or adjust the path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================
# HELPERS
# =============================================================

def make_black_image_bytes(size=(224, 224), color=(0, 0, 0), fmt="JPEG"):
    """Create a dummy in-memory image (no disk needed)."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf


def make_fake_model(predicted_index=2, n_classes=4):
    """
    Return a mock Keras model whose predict() returns a one-hot-like array.
    Default predicted_index=2 → 'notumor' (correct class order).
    """
    mock_model = MagicMock()
    probs = np.zeros((1, n_classes), dtype=np.float32)
    probs[0][predicted_index] = 0.95          # high confidence on chosen class
    mock_model.predict.return_value = probs
    return mock_model


# CLASS_NAMES as used in training (alphabetical / sorted)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']


# =============================================================
# FIXTURES
# =============================================================

@pytest.fixture
def client():
    """
    Spin up the Flask test client with the real model replaced
    by a mock so we don't need the .h5 file during testing.
    """
    with patch("main.load_model", return_value=make_fake_model(predicted_index=2)):
        from main import app
        app.config["TESTING"] = True
        app.config["UPLOAD_FOLDER"] = "test_uploads"
        os.makedirs("test_uploads", exist_ok=True)
        with app.test_client() as c:
            yield c


@pytest.fixture(autouse=True)
def cleanup_uploads():
    """Remove test upload files after every test."""
    yield
    import shutil
    if os.path.exists("test_uploads"):
        shutil.rmtree("test_uploads")


# =============================================================
# 1. ENDPOINT TESTS
# =============================================================

class TestEndpoints:

    def test_home_page_loads(self, client):
        """GET / should return 200 and HTML."""
        res = client.get("/")
        assert res.status_code == 200
        assert b"Tumor" in res.data or b"MRI" in res.data

    def test_home_page_has_upload_form(self, client):
        """Upload form must be present on the home page."""
        res = client.get("/")
        assert b'enctype="multipart/form-data"' in res.data
        assert b'type="file"' in res.data

    def test_invalid_route_returns_404(self, client):
        """Unknown routes should return 404."""
        res = client.get("/nonexistent-page")
        assert res.status_code == 404

    def test_post_without_file_key(self, client):
        """POST with no file field should not crash (graceful handling)."""
        res = client.post("/", data={}, content_type="multipart/form-data")
        assert res.status_code == 200
        assert b"No file" in res.data


# =============================================================
# 2. FILE UPLOAD TESTS
# =============================================================

class TestFileUpload:

    def test_upload_valid_jpeg(self, client):
        """Valid JPEG upload should return 200 with a result."""
        img_bytes = make_black_image_bytes(fmt="JPEG")
        res = client.post(
            "/",
            data={"file": (img_bytes, "test.jpg")},
            content_type="multipart/form-data"
        )
        assert res.status_code == 200
        assert b"%" in res.data        # confidence score rendered

    def test_upload_valid_png(self, client):
        """Valid PNG upload should also work."""
        img_bytes = make_black_image_bytes(fmt="PNG")
        res = client.post(
            "/",
            data={"file": (img_bytes, "test.png")},
            content_type="multipart/form-data"
        )
        assert res.status_code == 200

    def test_upload_empty_filename(self, client):
        """Empty filename string should be rejected gracefully."""
        res = client.post(
            "/",
            data={"file": (io.BytesIO(b""), "")},
            content_type="multipart/form-data"
        )
        assert res.status_code == 200
        assert b"No file" in res.data

    def test_uploaded_file_is_served(self, client):
        """Uploaded image should be accessible at /uploads/<filename>."""
        img_bytes = make_black_image_bytes(fmt="JPEG")
        client.post(
            "/",
            data={"file": (img_bytes, "serve_test.jpg")},
            content_type="multipart/form-data"
        )
        res = client.get("/uploads/serve_test.jpg")
        assert res.status_code == 200

    def test_large_image_handled(self, client):
        """A larger image (512x512) should still be handled without error."""
        img_bytes = make_black_image_bytes(size=(512, 512), fmt="JPEG")
        res = client.post(
            "/",
            data={"file": (img_bytes, "large.jpg")},
            content_type="multipart/form-data"
        )
        assert res.status_code == 200


# =============================================================
# 3. PREDICTION / CLASS LABEL TESTS
# =============================================================

class TestPredictions:

    def _upload_with_model(self, client, predicted_index):
        """Helper: upload a dummy image with a specific predicted class."""
        with patch("main.model", make_fake_model(predicted_index)):
            img_bytes = make_black_image_bytes(fmt="JPEG")
            return client.post(
                "/",
                data={"file": (img_bytes, "img.jpg")},
                content_type="multipart/form-data"
            )

    def test_predicts_notumor(self, client):
        """Index 2 → 'notumor' → response must say 'No Tumor'."""
        res = self._upload_with_model(client, predicted_index=2)
        assert b"No Tumor" in res.data

    def test_predicts_glioma(self, client):
        """Index 0 → 'glioma' → response must say 'Tumor'."""
        res = self._upload_with_model(client, predicted_index=0)
        assert b"Tumor" in res.data
        assert b"Glioma" in res.data

    def test_predicts_meningioma(self, client):
        """Index 1 → 'meningioma' → response must contain 'Meningioma'."""
        res = self._upload_with_model(client, predicted_index=1)
        assert b"Meningioma" in res.data

    def test_predicts_pituitary(self, client):
        """Index 3 → 'pituitary' → response must contain 'Pituitary'."""
        res = self._upload_with_model(client, predicted_index=3)
        assert b"Pituitary" in res.data

    def test_confidence_displayed(self, client):
        """Confidence percentage must appear in the response."""
        res = self._upload_with_model(client, predicted_index=2)
        assert b"95.00%" in res.data or b"%" in res.data

    def test_class_label_order_is_correct(self):
        """
        CRITICAL: Verify class_labels in main.py matches training order.
        Training uses sorted(os.listdir()) = alphabetical order.
        """
        from main import class_labels
        expected = ['glioma', 'meningioma', 'notumor', 'pituitary']
        assert class_labels == expected, (
            f"\n❌ CLASS ORDER MISMATCH!\n"
            f"   main.py has : {class_labels}\n"
            f"   Expected    : {expected}\n"
            f"   This causes notumor ↔ pituitary swaps!"
        )


# =============================================================
# 4. END-TO-END TESTS
# =============================================================

class TestEndToEnd:

    def test_full_flow_notumor(self, client):
        """
        Full flow: upload image → model predicts notumor →
        page shows result + confidence + image path.
        """
        with patch("main.model", make_fake_model(predicted_index=2)):
            img_bytes = make_black_image_bytes(fmt="JPEG")
            res = client.post(
                "/",
                data={"file": (img_bytes, "mri_notumor.jpg")},
                content_type="multipart/form-data"
            )
            html = res.data.decode("utf-8")
            assert res.status_code == 200
            assert "No Tumor" in html
            assert "%" in html                          # confidence shown
            assert "mri_notumor.jpg" in html            # image path shown

    def test_full_flow_glioma(self, client):
        """Full flow for glioma prediction."""
        with patch("main.model", make_fake_model(predicted_index=0)):
            img_bytes = make_black_image_bytes(fmt="JPEG")
            res = client.post(
                "/",
                data={"file": (img_bytes, "mri_glioma.jpg")},
                content_type="multipart/form-data"
            )
            html = res.data.decode("utf-8")
            assert "Glioma" in html

    def test_uploaded_image_accessible_after_prediction(self, client):
        """After prediction, uploaded image must be servable at /uploads/."""
        with patch("main.model", make_fake_model(predicted_index=3)):
            img_bytes = make_black_image_bytes(fmt="JPEG")
            client.post(
                "/",
                data={"file": (img_bytes, "e2e_check.jpg")},
                content_type="multipart/form-data"
            )
            img_res = client.get("/uploads/e2e_check.jpg")
            assert img_res.status_code == 200

    def test_multiple_sequential_uploads(self, client):
        """App must handle multiple uploads in a row without crashing."""
        filenames = ["scan1.jpg", "scan2.jpg", "scan3.jpg"]
        for fname in filenames:
            with patch("main.model", make_fake_model(predicted_index=2)):
                img_bytes = make_black_image_bytes(fmt="JPEG")
                res = client.post(
                    "/",
                    data={"file": (img_bytes, fname)},
                    content_type="multipart/form-data"
                )
                assert res.status_code == 200
