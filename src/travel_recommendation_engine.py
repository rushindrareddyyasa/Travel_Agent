
"""
Travel Recommendation Engine
=============================

Reusable recommendation engine for the Travel Agent project.

This module is intended to be used by the application layer.
The recommendation artifacts are loaded from:

models/recommendation_engine/

The recommendation logic is based on the validated engine
developed in Notebook 09.
"""

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd


# =========================================================
# PROJECT / ARTIFACT PATHS
# =========================================================

MODULE_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = MODULE_DIR.parent

ENGINE_DIR = (
    PROJECT_ROOT
    / "models"
    / "recommendation_engine"
)


# =========================================================
# ARTIFACT LOADER
# =========================================================

def _load_pickle(filename):
    """Load one saved pickle artifact."""

    path = ENGINE_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"Required engine artifact not found: {path}"
        )

    with open(path, "rb") as file:
        return pickle.load(file)


def _load_json(filename):
    """Load one saved JSON configuration file."""

    path = ENGINE_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"Required engine configuration not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


# =========================================================
# LOAD SAVED ENGINE ARTIFACTS
# =========================================================

processed_df = _load_pickle(
    "processed_df.pkl"
)

normalized_features = _load_pickle(
    "normalized_features.pkl"
)

destination_profile_scores = _load_pickle(
    "destination_profile_scores.pkl"
)

destination_similarity = _load_pickle(
    "destination_similarity.pkl"
)

destination_order = _load_pickle(
    "destination_order.pkl"
)

availability = _load_pickle(
    "availability.pkl"
)

model_features = _load_pickle(
    "model_features.pkl"
)

feature_scaler = _load_pickle(
    "feature_scaler.pkl"
)

pca_model = _load_pickle(
    "pca_model.pkl"
)

recommendation_config = _load_json(
    "recommendation_config.json"
)

feature_metadata = _load_json(
    "feature_metadata.json"
)

artifact_manifest = _load_json(
    "artifact_manifest.json"
)


# =========================================================
# TRUSTED DESTINATION ORDER
# =========================================================

if isinstance(
    destination_order,
    pd.DataFrame
):

    destinations = (
        destination_order[
            "destination"
        ]
        .astype(str)
        .tolist()
    )

else:

    destinations = list(
        destination_order
    )


# =========================================================
# CONFIGURATION
# =========================================================

preference_groups = (
    recommendation_config[
        "preference_groups"
    ]
)

preference_directions = (
    recommendation_config[
        "preference_directions"
    ]
)

interest_profiles = (
    recommendation_config[
        "interest_profiles"
    ]
)

travel_factor_weight = float(
    recommendation_config[
        "travel_factor_weight"
    ]
)

interest_weight = float(
    recommendation_config[
        "interest_weight"
    ]
)

hybrid_preference_weight = float(
    recommendation_config[
        "hybrid_preference_weight"
    ]
)

hybrid_similarity_weight = float(
    recommendation_config[
        "hybrid_similarity_weight"
    ]
)


# =========================================================
# USER PROFILE FIELDS
# =========================================================

USER_PREFERENCE_FIELDS = list(
    preference_groups.keys()
)

USER_INTEREST_FIELDS = list(
    interest_profiles
)


# =========================================================
# DESTINATION ALIGNMENT
# =========================================================

def _align_scores(scores):
    """
    Align a score Series to the trusted destination order.
    """

    if isinstance(
        scores,
        pd.Series
    ):

        if set(scores.index) == set(
            destinations
        ):

            aligned = scores.reindex(
                destinations
            )

        elif len(scores) == len(
            destinations
        ):

            aligned = pd.Series(
                scores.to_numpy(),
                index=destinations,
                dtype=float
            )

        else:

            raise ValueError(
                "Score count does not match "
                "destination count."
            )

    else:

        values = np.asarray(
            scores,
            dtype=float
        )

        if len(values) != len(
            destinations
        ):

            raise ValueError(
                "Score count does not match "
                "destination count."
            )

        aligned = pd.Series(
            values,
            index=destinations,
            dtype=float
        )

    return aligned


# =========================================================
# USER PROFILE VALIDATION
# =========================================================

def validate_user_profile(
    user_preferences,
    user_interests
):
    """
    Validate application user input.

    Values must be numeric and within [0, 1].
    """

    for field in USER_PREFERENCE_FIELDS:

        if field not in user_preferences:

            raise ValueError(
                f"Missing preference field: {field}"
            )

        value = float(
            user_preferences[field]
        )

        if not np.isfinite(value):

            raise ValueError(
                f"Invalid value for {field}"
            )

        if not 0.0 <= value <= 1.0:

            raise ValueError(
                f"{field} must be between 0 and 1."
            )


    for field in USER_INTEREST_FIELDS:

        if field not in user_interests:

            raise ValueError(
                f"Missing interest field: {field}"
            )

        value = float(
            user_interests[field]
        )

        if not np.isfinite(value):

            raise ValueError(
                f"Invalid value for {field}"
            )

        if not 0.0 <= value <= 1.0:

            raise ValueError(
                f"{field} must be between 0 and 1."
            )


# =========================================================
# PREFERENCE SCORING
# =========================================================

def calculate_preference_scores(
    user_preferences
):
    """
    Calculate personalized travel-preference scores.

    Missing feature values are ignored rather than treated
    as zero. Active weights are renormalized per destination.
    """

    weighted_total = pd.Series(
        0.0,
        index=destinations,
        dtype=float
    )

    active_weight_total = pd.Series(
        0.0,
        index=destinations,
        dtype=float
    )


    for group, features in (
        preference_groups.items()
    ):

        group_weight = float(
            user_preferences[group]
        )

        if group_weight == 0.0:
            continue


        for feature in features:

            values = (
                normalized_features[
                    feature
                ]
                .astype(float)
            )

            values = _align_scores(
                values
            )


            available_mask = (
                values.notna()
                &
                np.isfinite(
                    values.fillna(0.0)
                )
            )


            direction = (
                preference_directions[
                    feature
                ]
            )


            if direction == "higher":

                match_score = values

            elif direction == "lower":

                match_score = (
                    1.0 - values
                )

            else:

                raise ValueError(
                    f"Unknown preference direction "
                    f"for {feature}: {direction}"
                )


            weighted_total += (
                match_score
                .where(
                    available_mask,
                    0.0
                )
                *
                group_weight
            )


            active_weight_total += (
                available_mask.astype(float)
                *
                group_weight
            )


    result = pd.Series(
        0.0,
        index=destinations,
        dtype=float
    )


    valid_mask = (
        active_weight_total > 0.0
    )


    result.loc[valid_mask] = (
        weighted_total.loc[valid_mask]
        /
        active_weight_total.loc[valid_mask]
    )


    result.name = (
        "personalized_preference_score"
    )

    return result


# =========================================================
# INTEREST SCORING
# =========================================================

def calculate_interest_scores(
    user_interests
):
    """
    Calculate personalized interest scores.
    """

    weighted_total = pd.Series(
        0.0,
        index=destinations,
        dtype=float
    )

    active_weight = 0.0


    for interest in interest_profiles:

        weight = float(
            user_interests[interest]
        )

        if weight == 0.0:
            continue


        column = (
            f"{interest}_score"
        )


        values = (
            destination_profile_scores[
                column
            ]
            .astype(float)
        )


        if (
            "destination"
            in destination_profile_scores.columns
        ):

            values.index = (
                destination_profile_scores[
                    "destination"
                ]
            )


        values = _align_scores(
            values
        )


        weighted_total += (
            values
            *
            weight
        )

        active_weight += weight


    if active_weight == 0.0:

        result = pd.Series(
            0.0,
            index=destinations,
            dtype=float
        )

    else:

        result = (
            weighted_total
            /
            active_weight
        )


    result.name = (
        "personalized_interest_score"
    )

    return result


# =========================================================
# SIMILARITY SCORING
# =========================================================

def calculate_similarity_scores(
    reference_destination
):
    """
    Calculate normalized similarity scores relative to a
    reference destination.
    """

    if reference_destination not in destinations:

        raise ValueError(
            f"Unknown reference destination: "
            f"{reference_destination}"
        )


    raw_similarity = (
        destination_similarity
        .loc[
            reference_destination
        ]
        .reindex(
            destinations
        )
        .astype(float)
    )


    similarity_min = (
        raw_similarity.min()
    )

    similarity_max = (
        raw_similarity.max()
    )


    if similarity_max == similarity_min:

        result = pd.Series(
            0.5,
            index=destinations,
            dtype=float
        )

    else:

        result = (
            raw_similarity
            -
            similarity_min
        ) / (
            similarity_max
            -
            similarity_min
        )


    result.name = (
        "similarity_score"
    )

    return result


# =========================================================
# MAIN RECOMMENDATION FUNCTION
# =========================================================

def get_recommendations(
    user_preferences,
    user_interests,
    reference_destination=None,
    mode="hybrid",
    top_k=10
):
    """
    Generate ranked travel recommendations.

    Supported modes:
        preference
        similarity
        hybrid
    """

    valid_modes = [
        "preference",
        "similarity",
        "hybrid"
    ]


    if mode not in valid_modes:

        raise ValueError(
            f"Invalid mode '{mode}'. "
            f"Choose from {valid_modes}."
        )


    if top_k <= 0:

        raise ValueError(
            "top_k must be greater than zero."
        )


    validate_user_profile(
        user_preferences,
        user_interests
    )


    if mode in [
        "similarity",
        "hybrid"
    ]:

        if reference_destination is None:

            raise ValueError(
                "reference_destination is required "
                f"for {mode} mode."
            )


    preference_scores = (
        calculate_preference_scores(
            user_preferences
        )
    )


    interest_scores = (
        calculate_interest_scores(
            user_interests
        )
    )


    personalized_scores = (

        preference_scores
        *
        travel_factor_weight

        +

        interest_scores
        *
        interest_weight
    )


    if mode in [
        "similarity",
        "hybrid"
    ]:

        similarity_scores = (
            calculate_similarity_scores(
                reference_destination
            )
        )

    else:

        similarity_scores = pd.Series(
            0.0,
            index=destinations,
            dtype=float
        )


    if mode == "preference":

        final_scores = (
            personalized_scores
        )

    elif mode == "similarity":

        final_scores = (
            similarity_scores
        )

    else:

        final_scores = (

            personalized_scores
            *
            hybrid_preference_weight

            +

            similarity_scores
            *
            hybrid_similarity_weight
        )


    result = pd.DataFrame({

        "destination":
            destinations,

        "personalized_preference_score":
            preference_scores.values,

        "personalized_interest_score":
            interest_scores.values,

        "similarity_score":
            similarity_scores.values,

        "final_recommendation_score":
            final_scores.values
    })


    if reference_destination is not None:

        result = result[
            result[
                "destination"
            ]
            != reference_destination
        ]


    result = (
        result
        .sort_values(
            "final_recommendation_score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    result.insert(
        0,
        "rank",
        range(
            1,
            len(result) + 1
        )
    )


    return result.head(
        min(
            top_k,
            len(result)
        )
    )


# =========================================================
# SIMPLE EXPLANATION FUNCTION
# =========================================================

def explain_recommendation(
    recommendation_row,
    reference_destination=None
):
    """
    Create a human-readable explanation for a
    recommendation result.
    """

    destination = recommendation_row[
        "destination"
    ]

    preference_score = float(
        recommendation_row[
            "personalized_preference_score"
        ]
    )

    interest_score = float(
        recommendation_row[
            "personalized_interest_score"
        ]
    )

    similarity_score = float(
        recommendation_row[
            "similarity_score"
        ]
    )

    final_score = float(
        recommendation_row[
            "final_recommendation_score"
        ]
    )


    reasons = []


    if preference_score >= 0.70:

        reasons.append(
            "strongly matches your travel preferences"
        )

    elif preference_score >= 0.45:

        reasons.append(
            "matches your travel preferences well"
        )

    else:

        reasons.append(
            "has some alignment with your travel preferences"
        )


    if interest_score >= 0.70:

        reasons.append(
            "strongly matches your interests"
        )

    elif interest_score >= 0.45:

        reasons.append(
            "matches your interests well"
        )

    else:

        reasons.append(
            "has some alignment with your interests"
        )


    if reference_destination is not None:

        if similarity_score >= 0.70:

            reasons.append(
                f"is highly similar to {reference_destination}"
            )

        elif similarity_score >= 0.45:

            reasons.append(
                f"has good similarity to {reference_destination}"
            )


    if len(reasons) == 1:

        explanation_text = (
            f"{destination} was recommended because it "
            f"{reasons[0]}."
        )

    else:

        explanation_text = (
            f"{destination} was recommended because it "
            +
            ", ".join(
                reasons[:-1]
            )
            +
            ", and "
            +
            reasons[-1]
            +
            "."
        )


    return {

        "destination":
            destination,

        "final_score":
            final_score,

        "preference_score":
            preference_score,

        "interest_score":
            interest_score,

        "similarity_score":
            similarity_score,

        "explanation":
            explanation_text
    }


# =========================================================
# FULL EXPLANATION
# =========================================================

def get_full_recommendation_explanation(
    recommendations,
    reference_destination=None
):
    """
    Generate explanations for all recommendation rows.
    """

    explanations = []

    for _, row in recommendations.iterrows():

        explanations.append(
            explain_recommendation(
                row,
                reference_destination
            )
        )

    return explanations
