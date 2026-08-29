
import streamlit as st
import pandas as pd
import sys
from pathlib import Path


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(SRC_DIR)
    )


# =========================================================
# FASTAPI BACKEND CONFIGURATION
# =========================================================
#
# Streamlit is now the FRONTEND.
# Recommendation generation happens through the deployed
# FastAPI backend on Render.
#
# The API URL can be overridden with the
# TRAVEL_AGENT_API_URL environment variable.
# =========================================================

import os
import requests

API_URL = os.getenv(
    "TRAVEL_AGENT_API_URL",
    "https://travel-agent-api-t66t.onrender.com"
).rstrip("/")

RECOMMEND_ENDPOINT = f"{API_URL}/recommend"


# =========================================================
# LOAD DESTINATION DATA FOR UI DETAILS
# =========================================================
#
# We do NOT import the recommendation engine here.
# The ML recommendation calculation is handled by FastAPI.
#
# The Streamlit UI still needs destination names and travel
# details such as flights, hotels and weather. These are read
# directly from the project's cleaned dataset.
# =========================================================

def load_destination_data():
    """Load the cleaned destination dataset used for UI details."""

    candidate_files = [
        PROJECT_ROOT / "data" / "cleaned" / "integrated_travel_dataset.csv",
        PROJECT_ROOT / "data" / "cleaned" / "travel_integrated_preprocessed.csv",
        PROJECT_ROOT / "data" / "cleaned" / "travel_features.csv",
    ]

    for file_path in candidate_files:

        if file_path.exists():

            try:

                dataframe = pd.read_csv(file_path)

                # The application expects a destination column.
                if "destination" in dataframe.columns:

                    return dataframe

                # Some datasets may use a capitalized column name.
                if "Destination" in dataframe.columns:

                    dataframe = dataframe.rename(
                        columns={"Destination": "destination"}
                    )

                    return dataframe

            except Exception:
                continue

    return pd.DataFrame()


processed_df = load_destination_data()

if not processed_df.empty:

    destinations = sorted(
        processed_df["destination"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

else:

    destinations = []


# =========================================================
# FASTAPI REQUEST HELPER
# =========================================================

def get_recommendations_from_api(
    preferences,
    interests,
    reference_destination,
    mode,
    top_k
):
    """
    Send the user's Streamlit inputs to the deployed FastAPI API.

    FastAPI performs the actual recommendation calculation
    using the saved ML/recommendation artifacts.
    """

    payload = {
        "reference_destination": reference_destination,
        "mode": mode,
        "top_k": top_k,
        "preferences": preferences,
        "interests": interests
    }

    try:

        response = requests.post(
            RECOMMEND_ENDPOINT,
            json=payload,
            timeout=120
        )

        # Convert HTTP errors such as 422/500 into exceptions.
        response.raise_for_status()

        return response.json()

    except requests.exceptions.Timeout:

        st.error(
            "The Travel Agent API took too long to respond. "
            "Please try again."
        )

    except requests.exceptions.RequestException as error:

        st.error(
            f"Unable to connect to the Travel Agent API: {error}"
        )

    except ValueError:

        st.error(
            "The Travel Agent API returned an invalid response."
        )

    return None


# =========================================================
# API RESPONSE → DATAFRAME
# =========================================================

def api_response_to_dataframe(api_response):
    """
    Convert the FastAPI JSON recommendation response into
    the DataFrame format expected by the existing UI.
    """

    if not api_response:

        return pd.DataFrame()

    recommendation_rows = api_response.get(
        "recommendations",
        []
    )

    if not recommendation_rows:

        return pd.DataFrame()

    return pd.DataFrame(
        recommendation_rows
    )


# =========================================================
# RECOMMENDATION EXPLANATION
# =========================================================

def explain_recommendation(row, reference_destination):
    """
    Create a human-readable explanation from the scores returned
    by FastAPI.

    The scoring itself is NOT performed here. FastAPI has already
    calculated the three component scores and the final score.
    """

    preference_score = float(
        row["personalized_preference_score"]
    )

    interest_score = float(
        row["personalized_interest_score"]
    )

    similarity_score = float(
        row["similarity_score"]
    )

    # Describe the strongest reason for the recommendation.
    score_map = {
        "travel preferences": preference_score,
        "your interests": interest_score,
        f"similarity to {reference_destination}": similarity_score
    }

    strongest_reason = max(
        score_map,
        key=score_map.get
    )

    strongest_score = score_map[
        strongest_reason
    ]

    if strongest_score >= 0.70:

        strength_text = "strongly"

    elif strongest_score >= 0.50:

        strength_text = "well"

    else:

        strength_text = "to some extent"

    explanation = (
        f"{row['destination']} was recommended because it "
        f"matches {strongest_reason} {strength_text}."
    )

    return {
        "explanation": explanation
    }






# =========================================================
# UI VALUE FORMATTER
# =========================================================

def format_ui_value(value, decimals=2):
    """
    Convert raw dataset values into user-friendly display
    values without changing the underlying data.
    """

    # -----------------------------------------------------
    # Handle missing values.
    # -----------------------------------------------------

    if value is None:

        return "Not available"


    try:

        if pd.isna(value):

            return "Not available"

    except (TypeError, ValueError):

        pass


    # -----------------------------------------------------
    # Format numeric values.
    # -----------------------------------------------------

    if isinstance(
        value,
        (int, float)
    ):

        return f"{value:.{decimals}f}"


    return str(value)


# =========================================================
# DESTINATION DETAILS HELPER
# =========================================================

def get_destination_details(destination):
    """
    Retrieve useful travel information for a destination
    from the existing processed dataset.
    """

    matches = processed_df[
        processed_df["destination"].astype(str).str.strip()
        == str(destination).strip()
    ]


    if matches.empty:

        return {}


    row = matches.iloc[0]


    details = {}


    # -----------------------------------------------------
    # Flight information.
    # -----------------------------------------------------

    flight_fields = {

        "flight_count":
            "Available Flights",

        "avg_flight_price":
            "Average Flight Price",

        "avg_total_duration":
            "Average Flight Duration",

        "avg_outbound_stops":
            "Average Outbound Stops"
    }


    for field, label in flight_fields.items():

        if field in row.index:

            details[label] = row[field]


    # -----------------------------------------------------
    # Accommodation information.
    # -----------------------------------------------------

    accommodation_fields = {

        "hotel_count":
            "Hotels",

        "room_count":
            "Rooms",

        "min_hotel_price":
            "Minimum Hotel Price",

        "avg_hotel_price":
            "Average Hotel Price"
    }


    for field, label in accommodation_fields.items():

        if field in row.index:

            details[label] = row[field]


    # -----------------------------------------------------
    # Weather information.
    # -----------------------------------------------------

    weather_fields = {

        "temperature":
            "Temperature",

        "feels_like":
            "Feels Like",

        "humidity":
            "Humidity",

        "wind_speed":
            "Wind Speed"
    }


    for field, label in weather_fields.items():

        if field in row.index:

            details[label] = row[field]


    # -----------------------------------------------------
    # Destination characteristics.
    # -----------------------------------------------------

    characteristic_fields = {

        "sight_count":
            "Sights",

        "park_count":
            "Parks",

        "restaurant_count":
            "Restaurants",

        "water_count":
            "Water Features",

        "mountain_count":
            "Mountains",

        "coastal_count":
            "Coastal Features"
    }


    for field, label in characteristic_fields.items():

        if field in row.index:

            details[label] = row[field]


    return details


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(

    page_title="Travel Agent",

    page_icon="✈️",

    layout="wide",

    initial_sidebar_state="expanded"
)


# =========================================================
# APPLICATION HEADER
# =========================================================

st.title("✈️ Travel Agent")

st.subheader(
    "Personalized Destination Recommendation System"
)

st.write(
    """
    Discover destinations that match your travel
    preferences, interests, and similarity to a destination
    you already like.
    """
)


# =========================================================
# SIDEBAR — TRIP SETTINGS
# =========================================================

st.sidebar.header("⚙️ Trip Settings")


reference_destination = st.sidebar.selectbox(

    "📍 Reference Destination",

    options=destinations,

    index=(
        destinations.index("Goa")
        if "Goa" in destinations
        else 0
    ),

    help=(
        "Choose a destination whose characteristics "
        "you would like your recommendations to resemble."
    )
)


recommendation_mode = st.sidebar.selectbox(

    "Recommendation Mode",

    options=[
        "hybrid",
        "preference",
        "similarity"
    ],

    format_func=lambda mode: {

        "hybrid":
            "Hybrid — Recommended",

        "preference":
            "Preference Based",

        "similarity":
            "Similarity Based"

    }[mode],

    index=0
)


top_k = st.sidebar.selectbox(

    "Number of Recommendations",

    options=[
        5,
        10,
        15
    ],

    index=1
)


# =========================================================
# TRAVEL PREFERENCES
# =========================================================

st.header("🎯 Your Travel Preferences")

st.caption(
    "Rate how important each travel factor is to you."
)


preference_col1, preference_col2 = st.columns(2)


with preference_col1:

    budget = st.slider(

        "💰 Budget",

        0.0,

        1.0,

        0.5,

        0.1,

        help=(
            "Importance of keeping travel costs affordable."
        )
    )


    flight = st.slider(

        "✈️ Flight Convenience",

        0.0,

        1.0,

        0.5,

        0.1,

        help=(
            "Importance of convenient flight options."
        )
    )


    accommodation = st.slider(

        "🏨 Accommodation",

        0.0,

        1.0,

        0.5,

        0.1,

        help=(
            "Importance of accommodation availability "
            "and value."
        )
    )


with preference_col2:

    weather = st.slider(

        "🌤️ Weather",

        0.0,

        1.0,

        0.5,

        0.1,

        help=(
            "Importance of favorable weather."
        )
    )


    destination_characteristics = st.slider(

        "🏞️ Destination Characteristics",

        0.0,

        1.0,

        0.5,

        0.1,

        help=(
            "Importance of attractions and destination "
            "features."
        )
    )


# =========================================================
# DESTINATION INTERESTS
# =========================================================

st.header("❤️ Your Interests")

st.caption(
    "Tell the Travel Agent what you enjoy."
)


interest_col1, interest_col2 = st.columns(2)


with interest_col1:

    nature = st.slider(

        "🌿 Nature",

        0.0,

        1.0,

        0.5,

        0.1,

        help=(
            "Interest in natural landscapes and environments."
        )
    )


    sightseeing = st.slider(

        "🏛️ Sightseeing",

        0.0,

        1.0,

        0.5,

        0.1,

        help=(
            "Interest in attractions and sightseeing."
        )
    )


with interest_col2:

    water_coastal = st.slider(

        "🌊 Water & Coastal",

        0.0,

        1.0,

        0.5,

        0.1,

        help=(
            "Interest in beaches, rivers and coastal places."
        )
    )


    wildlife = st.slider(

        "🐅 Wildlife",

        0.0,

        1.0,

        0.5,

        0.1,

        help=(
            "Interest in wildlife and protected areas."
        )
    )


# =========================================================
# GENERATE RECOMMENDATIONS
# =========================================================

st.divider()


generate_button = st.button(

    "🔍 Find My Destinations",

    type="primary",

    use_container_width=True
)


if generate_button:

    # -----------------------------------------------------
    # Convert UI controls into the profile expected by the
    # existing recommendation engine.
    # -----------------------------------------------------

    user_preferences = {

        "budget":
            budget,

        "flight":
            flight,

        "accommodation":
            accommodation,

        "weather":
            weather,

        "destination_characteristics":
            destination_characteristics
    }


    user_interests = {

        "nature":
            nature,

        "sightseeing":
            sightseeing,

        "water_coastal":
            water_coastal,

        "wildlife":
            wildlife
    }


    # -----------------------------------------------------
    # Generate recommendations through the deployed FastAPI
    # backend instead of running the recommendation engine
    # directly inside Streamlit.
    # -----------------------------------------------------

    try:

        api_response = get_recommendations_from_api(

            preferences=
                user_preferences,

            interests=
                user_interests,

            reference_destination=
                reference_destination,

            mode=
                recommendation_mode,

            top_k=
                top_k
        )

        if api_response is not None:

            recommendations = api_response_to_dataframe(
                api_response
            )

            if recommendations.empty:

                st.warning(
                    "The API returned no recommendations."
                )

            else:

                # -------------------------------------------------
                # Store recommendations in session state.
                # -------------------------------------------------

                st.session_state[
                    "recommendations"
                ] = recommendations

                st.session_state[
                    "reference_destination"
                ] = reference_destination

    except Exception as error:

        st.error(
            f"Unable to generate recommendations: {error}"
        )


# =========================================================
# DISPLAY RECOMMENDATIONS
# =========================================================

if "recommendations" in st.session_state:

    recommendations = st.session_state[
        "recommendations"
    ]

    reference_destination = st.session_state[
        "reference_destination"
    ]


    st.divider()

    st.header(
        "🌟 Recommended Destinations"
    )


    st.write(
        f"Based on your preferences and "
        f"**{reference_destination}** as the reference destination."
    )


    # -----------------------------------------------------
    # Display each recommendation as a clean card.
    # -----------------------------------------------------

    for _, row in recommendations.iterrows():

        destination = row[
            "destination"
        ]

        final_score = float(
            row[
                "final_recommendation_score"
            ]
        )


        with st.container(
            border=True
        ):

            col1, col2 = st.columns(
                [4, 1]
            )


            with col1:

                st.subheader(
                    f"#{int(row['rank'])}  {destination}"
                )


                # -------------------------------------------------
                # Explain why this destination was recommended.
                # -------------------------------------------------

                explanation = (
                    explain_recommendation(
                        row,
                        reference_destination
                    )
                )


                # -------------------------------------------------
                # Human-readable explanation.
                # -------------------------------------------------

                st.write(
                    explanation["explanation"]
                )


                # -------------------------------------------------
                # Show the three existing recommendation
                # components separately.
                #
                # These values are already calculated by the
                # recommendation engine.
                # -------------------------------------------------

                explanation_col1, explanation_col2, explanation_col3 = (
                    st.columns(3)
                )


                with explanation_col1:

                    st.metric(

                        "🎯 Preference Match",

                        f"{float(row['personalized_preference_score']) * 100:.1f}%"
                    )


                with explanation_col2:

                    st.metric(

                        "❤️ Interest Match",

                        f"{float(row['personalized_interest_score']) * 100:.1f}%"
                    )


                with explanation_col3:

                    st.metric(

                        "📍 Similarity",

                        f"{float(row['similarity_score']) * 100:.1f}%"
                    )


            with col2:

                st.metric(

                    "Match Score",

                    f"{final_score * 100:.1f}%"
                )



            # -------------------------------------------------
            # Quick travel summary.
            #
            # These values come directly from the existing
            # destination dataset.
            # -------------------------------------------------

            summary_details = get_destination_details(
                destination
            )


            summary_columns = st.columns(5)


            with summary_columns[0]:

                flights = summary_details.get(
                    "Available Flights"
                )

                if flights is not None:

                    st.metric(
                        "✈️ Flights",
                        format_ui_value(
                            flights,
                            decimals=0
                        )
                    )


            with summary_columns[1]:

                hotels = summary_details.get(
                    "Hotels"
                )

                if hotels is not None:

                    st.metric(
                        "🏨 Hotels",
                        format_ui_value(
                            hotels,
                            decimals=0
                        )
                    )


            with summary_columns[2]:

                temperature = summary_details.get(
                    "Temperature"
                )

                if temperature is not None:

                    st.metric(
                        "🌡️ Temperature",
                        format_ui_value(
                            temperature,
                            decimals=1
                        )
                    )


            with summary_columns[3]:

                sights = summary_details.get(
                    "Sights"
                )

                if sights is not None:

                    st.metric(
                        "🏛️ Sights",
                        format_ui_value(
                            sights,
                            decimals=0
                        )
                    )


            with summary_columns[4]:

                water = summary_details.get(
                    "Water Features"
                )

                if water is not None:

                    st.metric(
                        "🌊 Water Features",
                        format_ui_value(
                            water,
                            decimals=0
                        )
                    )


            # -----------------------------------------------------
            # Destination information.
            # -----------------------------------------------------

            details = get_destination_details(
                destination
            )


            if details:

                with st.expander(
                    "📋 Destination Details"
                ):

                    detail_col1, detail_col2 = (
                        st.columns(2)
                    )


                    detail_items = list(
                        details.items()
                    )


                    midpoint = (
                        len(detail_items) + 1
                    ) // 2


                    with detail_col1:

                        for label, value in (
                            detail_items[:midpoint]
                        ):

                            if (
                                value is not None
                                and str(value) != "nan"
                            ):

                                if isinstance(
                                    value,
                                    float
                                ):

                                    value = round(
                                        value,
                                        2
                                    )


                                st.write(
                                    f"**{label}:** {value}"
                                )


                    with detail_col2:

                        for label, value in (
                            detail_items[midpoint:]
                        ):

                            if (
                                value is not None
                                and str(value) != "nan"
                            ):

                                if isinstance(
                                    value,
                                    float
                                ):

                                    value = round(
                                        value,
                                        2
                                    )


                                st.write(
                                    f"**{label}:** {value}"
                                )


    # -----------------------------------------------------
    # Detailed score table.
    #
    # This gives the user transparency into how the engine
    # scored each recommendation.
    # -----------------------------------------------------

    st.subheader(
        "📊 Recommendation Details"
    )


    display_columns = [

        "rank",

        "destination",

        "personalized_preference_score",

        "personalized_interest_score",

        "similarity_score",

        "final_recommendation_score"
    ]


    display_df = recommendations[
        display_columns
    ].copy()


    display_df = display_df.rename(

        columns={

            "personalized_preference_score":
                "Preference Score",

            "personalized_interest_score":
                "Interest Score",

            "similarity_score":
                "Similarity Score",

            "final_recommendation_score":
                "Final Score"
        }
    )


    st.dataframe(

        display_df,

        use_container_width=True,

        hide_index=True
    )
