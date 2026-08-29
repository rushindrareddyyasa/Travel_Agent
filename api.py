# =========================================================
# TRAVEL AGENT - FASTAPI BACKEND
# =========================================================
#
# This file exposes the EXISTING Travel Agent
# recommendation engine through REST API endpoints.
#
# IMPORTANT:
# - No ML model is trained here.
# - No recommendation logic is recreated here.
# - Notebook 09 is NOT modified.
# - Notebook 10 is NOT modified.
# - The existing recommendation engine is reused.
#
# Architecture:
#
#     Client / Frontend
#            |
#            v
#         FastAPI
#            |
#            v
#   recommendation_engine.py
#            |
#            v
#      Saved ML artifacts
#
# =========================================================


# =========================================================
# IMPORTS
# =========================================================

from typing import Any, Dict

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# =========================================================
# IMPORT EXISTING RECOMMENDATION ENGINE
# =========================================================
#
# The actual recommendation calculations remain inside:
#
# src/travel_recommendation_engine.py
#
# We are only calling that existing function here.
# =========================================================

from src.travel_recommendation_engine import (
    get_recommendations
)


# =========================================================
# CREATE FASTAPI APPLICATION
# =========================================================

app = FastAPI(

    title="Travel Agent Recommendation API",

    description=(
        "REST API for the ML-based "
        "Travel Agent destination recommendation system."
    ),

    version="1.0.0"
)


# =========================================================
# REQUEST SCHEMA
# =========================================================
#
# This defines the JSON input accepted by:
#
# POST /recommend
#
# =========================================================


class RecommendationRequest(BaseModel):

    # -----------------------------------------------------
    # Destination that the user already likes.
    # -----------------------------------------------------

    reference_destination: str = Field(

        default="Goa",

        description=(
            "Destination used as the similarity reference."
        )
    )


    # -----------------------------------------------------
    # Recommendation strategy.
    #
    # Supported:
    #     preference
    #     similarity
    #     hybrid
    # -----------------------------------------------------

    mode: str = Field(

        default="hybrid",

        description=(
            "Recommendation mode: "
            "preference, similarity, or hybrid."
        )
    )


    # -----------------------------------------------------
    # Number of recommendations requested.
    # -----------------------------------------------------

    top_k: int = Field(

        default=10,

        ge=1,

        le=15,

        description=(
            "Number of destinations to recommend."
        )
    )


    # -----------------------------------------------------
    # Travel preference weights.
    #
    # Example:
    #
    # {
    #     "budget": 0.8,
    #     "flight": 0.6,
    #     "accommodation": 0.7,
    #     "weather": 0.9,
    #     "destination_characteristics": 0.6
    # }
    # -----------------------------------------------------

    preferences: Dict[str, float] = Field(

        default_factory=dict,

        description=(
            "User travel preference weights."
        )
    )


    # -----------------------------------------------------
    # Destination interest weights.
    #
    # Example:
    #
    # {
    #     "nature": 0.9,
    #     "sightseeing": 0.7,
    #     "water_coastal": 0.8,
    #     "wildlife": 0.4
    # }
    # -----------------------------------------------------

    interests: Dict[str, float] = Field(

        default_factory=dict,

        description=(
            "User destination interest weights."
        )
    )


# =========================================================
# JSON SERIALIZATION HELPER
# =========================================================
#
# The ML engine uses NumPy and Pandas.
#
# Therefore it can return values such as:
#
#     numpy.int64
#     numpy.float64
#     numpy.bool_
#     numpy.ndarray
#     pandas.DataFrame
#     pandas.Series
#
# Standard JSON cannot directly serialize all of these.
#
# This function recursively converts them into normal
# Python JSON-compatible objects.
# =========================================================


def make_json_serializable(value: Any) -> Any:

    # -----------------------------------------------------
    # Handle None.
    # -----------------------------------------------------

    if value is None:

        return None


    # -----------------------------------------------------
    # Handle NumPy integer values.
    #
    # numpy.int64 -> Python int
    # -----------------------------------------------------

    if isinstance(value, np.integer):

        return int(value)


    # -----------------------------------------------------
    # Handle NumPy floating-point values.
    #
    # numpy.float64 -> Python float
    #
    # NaN and Infinity are converted to None because they
    # are not valid standard JSON numbers.
    # -----------------------------------------------------

    if isinstance(value, np.floating):

        numeric_value = float(value)

        if not np.isfinite(numeric_value):

            return None

        return numeric_value


    # -----------------------------------------------------
    # Handle normal Python float values.
    #
    # This also protects the API if the engine returns
    # normal float NaN/Infinity values.
    # -----------------------------------------------------

    if isinstance(value, float):

        if not np.isfinite(value):

            return None

        return value


    # -----------------------------------------------------
    # Handle NumPy boolean values.
    # -----------------------------------------------------

    if isinstance(value, np.bool_):

        return bool(value)


    # -----------------------------------------------------
    # Handle NumPy arrays.
    #
    # ndarray -> Python list
    # -----------------------------------------------------

    if isinstance(value, np.ndarray):

        return make_json_serializable(
            value.tolist()
        )


    # -----------------------------------------------------
    # Handle Pandas DataFrames.
    #
    # DataFrame -> list of dictionaries
    #
    # This makes the API robust if the recommendation
    # engine returns a DataFrame.
    # -----------------------------------------------------

    if isinstance(value, pd.DataFrame):

        records = value.to_dict(
            orient="records"
        )

        return make_json_serializable(records)


    # -----------------------------------------------------
    # Handle Pandas Series.
    #
    # Series -> Python list
    # -----------------------------------------------------

    if isinstance(value, pd.Series):

        return make_json_serializable(
            value.tolist()
        )


    # -----------------------------------------------------
    # Handle dictionaries.
    #
    # Convert every value recursively.
    # -----------------------------------------------------

    if isinstance(value, dict):

        return {

            str(key):
                make_json_serializable(item)

            for key, item in value.items()
        }


    # -----------------------------------------------------
    # Handle lists.
    # -----------------------------------------------------

    if isinstance(value, list):

        return [

            make_json_serializable(item)

            for item in value
        ]


    # -----------------------------------------------------
    # Handle tuples.
    # -----------------------------------------------------

    if isinstance(value, tuple):

        return [

            make_json_serializable(item)

            for item in value
        ]


    # -----------------------------------------------------
    # Handle sets.
    # -----------------------------------------------------

    if isinstance(value, set):

        return [

            make_json_serializable(item)

            for item in value
        ]


    # -----------------------------------------------------
    # Normal Python values can be returned directly.
    # -----------------------------------------------------

    return value


# =========================================================
# ROOT ENDPOINT
# =========================================================
#
# Used to confirm that the API is running.
# =========================================================


@app.get("/")
def root():

    return {

        "service":
            "Travel Agent Recommendation API",

        "version":
            "1.0.0",

        "status":
            "running",

        "documentation":
            "/docs"
    }


# =========================================================
# HEALTH CHECK ENDPOINT
# =========================================================
#
# Useful for:
#
# - Local testing
# - Cloud deployment
# - Monitoring
# - Deployment health checks
#
# =========================================================


@app.get("/health")
def health_check():

    return {

        "status":
            "healthy",

        "service":
            "travel-agent-recommendation-api"
    }


# =========================================================
# RECOMMENDATION ENDPOINT
# =========================================================
#
# This is the main ML inference endpoint.
#
# The request is passed directly to the existing:
#
#     get_recommendations()
#
# function.
#
# =========================================================


@app.post("/recommend")
def recommend(
    request: RecommendationRequest
):

    # -----------------------------------------------------
    # Allowed recommendation modes.
    # -----------------------------------------------------

    allowed_modes = {

        "preference",

        "similarity",

        "hybrid"
    }


    # -----------------------------------------------------
    # Normalize the mode to lowercase.
    # -----------------------------------------------------

    mode = request.mode.lower()


    # -----------------------------------------------------
    # Validate recommendation mode.
    # -----------------------------------------------------

    if mode not in allowed_modes:

        raise HTTPException(

            status_code=400,

            detail=(
                "Invalid recommendation mode. "
                "Choose one of: "
                "preference, similarity, hybrid."
            )
        )


    try:

        # -------------------------------------------------
        # Call the existing recommendation engine.
        #
        # Exact confirmed signature:
        #
        # get_recommendations(
        #     user_preferences,
        #     user_interests,
        #     reference_destination=None,
        #     mode="hybrid",
        #     top_k=10
        # )
        # -------------------------------------------------

        recommendations = get_recommendations(

            user_preferences=
                request.preferences,

            user_interests=
                request.interests,

            reference_destination=
                request.reference_destination,

            mode=
                mode,

            top_k=
                request.top_k
        )


        # -------------------------------------------------
        # Convert NumPy/Pandas values returned by the
        # recommendation engine into standard Python
        # JSON-compatible values.
        # -------------------------------------------------

        recommendations = make_json_serializable(
            recommendations
        )


        # -------------------------------------------------
        # Return the API response.
        # -------------------------------------------------

        return {

            "reference_destination":
                request.reference_destination,

            "mode":
                mode,

            "top_k":
                request.top_k,

            "recommendations":
                recommendations
        }


    except HTTPException:

        # -------------------------------------------------
        # Preserve intentional HTTP errors.
        # -------------------------------------------------

        raise


    except Exception as error:

        # -------------------------------------------------
        # Handle unexpected recommendation engine errors.
        # -------------------------------------------------

        raise HTTPException(

            status_code=500,

            detail=(
                "Recommendation generation failed: "
                f"{error}"
            )
        )


# =========================================================
# END OF API
# =========================================================