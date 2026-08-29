"""
Travel Agent API Tests
======================

These tests verify the basic functionality of the FastAPI
backend used by the Travel Agent application.

Important:
- These tests do NOT train the ML model.
- These tests do NOT modify any model artifacts.
- They only verify that the API can receive requests and
  return the expected responses.
"""

from fastapi.testclient import TestClient

from api import app


# =========================================================
# CREATE FASTAPI TEST CLIENT
# =========================================================
#
# TestClient allows us to call the FastAPI application
# locally without starting a separate web server.
# =========================================================

client = TestClient(app)


# =========================================================
# TEST 1 — ROOT ENDPOINT
# =========================================================

def test_root_endpoint():
    """
    Verify that the root API endpoint is available.
    """

    response = client.get("/")

    # The API should respond successfully.
    assert response.status_code == 200

    data = response.json()

    # Verify the API reports that it is running.
    assert data["status"] == "running"


# =========================================================
# TEST 2 — HEALTH ENDPOINT
# =========================================================

def test_health_endpoint():
    """
    Verify that the API health-check endpoint works.
    """

    response = client.get("/health")

    # Health endpoint should return HTTP 200.
    assert response.status_code == 200

    data = response.json()

    # Verify the expected health status.
    assert data["status"] == "healthy"


# =========================================================
# TEST 3 — VALID RECOMMENDATION REQUEST
# =========================================================
#
# This is the most important API test.
#
# We send the same type of information that the Streamlit
# frontend sends to the FastAPI backend.
# =========================================================

def test_recommendation_endpoint():
    """
    Verify that /recommend accepts a valid request and
    returns destination recommendations.
    """

    payload = {

        # Destination used for similarity calculation.
        "reference_destination": "Goa",

        # Use the hybrid recommendation strategy.
        "mode": "hybrid",

        # Request five destinations.
        "top_k": 5,

        # User travel preferences.
        "preferences": {

            "budget": 0.5,

            "flight": 0.5,

            "accommodation": 0.5,

            "weather": 0.5,

            "destination_characteristics": 0.5
        },

        # User destination interests.
        "interests": {

            "nature": 0.5,

            "sightseeing": 0.5,

            "water_coastal": 0.5,

            "wildlife": 0.5
        }
    }


    # Send the request to the recommendation endpoint.
    response = client.post(
        "/recommend",
        json=payload
    )


    # The recommendation request should succeed.
    assert response.status_code == 200


    # Convert the response to a Python dictionary.
    data = response.json()


    # -----------------------------------------------------
    # Verify response structure.
    # -----------------------------------------------------

    assert data["reference_destination"] == "Goa"

    assert data["mode"] == "hybrid"

    assert data["top_k"] == 5


    # The API should return a recommendations list.
    assert "recommendations" in data

    assert isinstance(
        data["recommendations"],
        list
    )


    # We requested five recommendations.
    assert len(
        data["recommendations"]
    ) == 5


    # -----------------------------------------------------
    # Verify the structure of one recommendation.
    # -----------------------------------------------------

    first_recommendation = data[
        "recommendations"
    ][0]


    assert "rank" in first_recommendation

    assert "destination" in first_recommendation

    assert "personalized_preference_score" in first_recommendation

    assert "personalized_interest_score" in first_recommendation

    assert "similarity_score" in first_recommendation

    assert "final_recommendation_score" in first_recommendation


# =========================================================
# TEST 4 — REQUEST VALIDATION
# =========================================================
#
# top_k is restricted by the API to:
#
#     1 <= top_k <= 15
#
# Sending top_k = 0 should therefore produce HTTP 422.
# =========================================================

def test_recommendation_validation():
    """
    Verify that invalid top_k values are rejected by FastAPI.
    """

    payload = {

        "reference_destination": "Goa",

        "mode": "hybrid",

        # Invalid because the API requires top_k >= 1.
        "top_k": 0,

        "preferences": {

            "budget": 0.5,

            "flight": 0.5,

            "accommodation": 0.5,

            "weather": 0.5,

            "destination_characteristics": 0.5
        },

        "interests": {

            "nature": 0.5,

            "sightseeing": 0.5,

            "water_coastal": 0.5,

            "wildlife": 0.5
        }
    }


    response = client.post(
        "/recommend",
        json=payload
    )


    # FastAPI/Pydantic should reject the invalid value.
    assert response.status_code == 422