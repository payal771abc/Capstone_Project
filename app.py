# ============================================================
# WOMEN'S CAREER RECOMMENDATION SYSTEM
# STREAMLIT APPLICATION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Women's Career Recommendation System",
    page_icon="👩‍💼",
    layout="wide"
)


# ============================================================
# LOAD SAVED MODELS
# ============================================================

@st.cache_resource
def load_models():

    model = joblib.load(
        "women_career_model.pkl"
    )

    mlb_dict = joblib.load(
        "multi_label_binarizers.pkl"
    )

    return model, mlb_dict


model, mlb_dict = load_models()


# ============================================================
# TITLE
# ============================================================

st.title(
    "👩‍💼 Women's Career Recommendation System"
)

st.write(
    """
    This Machine Learning system recommends a suitable
    career based on education, skills, interests,
    experience, personality, salary expectation,
    location preference and work-life balance.
    """
)


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "📌 About Project"
    )

    st.write(
        """
        The system uses Machine Learning to recommend
        suitable careers based on a woman's profile.
        """
    )

    st.subheader(
        "How it works?"
    )

    st.write(
        """
        1. Enter your details

        2. Select skills and interests

        3. Click Recommend Career

        4. Get the predicted career
        """
    )

    st.info(
        "Main ML Task: Multi-Class Classification"
    )


# ============================================================
# USER INPUT
# ============================================================

st.header(
    "📝 Enter Your Details"
)


col1, col2 = st.columns(2)


# ============================================================
# COLUMN 1
# ============================================================

with col1:

    # Age
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=60,
        value=22,
        step=1
    )


    # Education
    education = st.selectbox(
        "Education",
        [
            "12th",
            "Diploma",
            "Undergraduate",
            "Postgraduate"
        ]
    )


    # Degree
    degree = st.selectbox(
        "Degree",
        [
            "12th",
            "Diploma in Computer",
            "Diploma in Management",
            "Diploma in Design",
            "BCA",
            "B.Tech",
            "B.Sc",
            "BBA",
            "BA",
            "B.Com",
            "MCA",
            "M.Tech",
            "M.Sc",
            "MBA",
            "MA",
            "M.Com"
        ]
    )


    # Skills
    skills = st.multiselect(
        "Skills",
        [
            "Communication",
            "Programming",
            "Analysis",
            "Leadership",
            "Research",
            "Creativity",
            "Management",
            "Problem Solving"
        ]
    )


    # Technical Skills
    technical_skills = st.multiselect(
        "Technical Skills",
        [
            "Python",
            "SQL",
            "Machine Learning",
            "Deep Learning",
            "TensorFlow",
            "Java",
            "C++",
            "Git",
            "Excel",
            "Power BI",
            "Networking",
            "Linux",
            "Cybersecurity",
            "Digital Marketing",
            "SEO",
            "Google Analytics",
            "Figma",
            "Adobe XD",
            "UI Design",
            "Prototyping",
            "Statistics",
            "Research Methods",
            "SPSS",
            "Business Analysis",
            "HR Analytics",
            "MS Office"
        ]
    )


    # Soft Skills
    soft_skills = st.multiselect(
        "Soft Skills",
        [
            "Communication",
            "Leadership",
            "Teamwork",
            "Problem Solving",
            "Time Management",
            "Critical Thinking",
            "Adaptability",
            "Creativity"
        ]
    )


    # Work Experience
    work_experience = st.number_input(
        "Work Experience (Years)",
        min_value=0,
        max_value=30,
        value=0,
        step=1
    )


# ============================================================
# COLUMN 2
# ============================================================

with col2:

    # Interests
    interests = st.multiselect(
        "Interests",
        [
            "Data Science",
            "Analytics",
            "AI",
            "Data Analysis",
            "Business Intelligence",
            "Machine Learning",
            "Technology",
            "Programming",
            "Software Development",
            "Cybersecurity",
            "Networking",
            "Human Resources",
            "People Management",
            "Recruitment",
            "Marketing",
            "Branding",
            "Social Media",
            "Research",
            "Education",
            "Academics",
            "Design",
            "Creativity",
            "User Experience",
            "Business",
            "Management"
        ]
    )


    # Career Interest
    career_interest = st.selectbox(
        "Career Interests",
        [
            "Data Science",
            "Data Analytics",
            "Artificial Intelligence",
            "Software Development",
            "Cybersecurity",
            "Human Resources",
            "Marketing",
            "Research",
            "UI/UX Design",
            "Business Analysis"
        ]
    )


    # Personality
    personality = st.multiselect(
        "Personality / Strengths",
        [
            "Analytical",
            "Problem Solving",
            "Curious",
            "Detail Oriented",
            "Innovative",
            "Logical",
            "Creative",
            "Empathetic",
            "Leadership",
            "Communication",
            "Alert"
        ]
    )


    # Work Environment
    work_environment = st.selectbox(
        "Preferred Work Environment",
        [
            "Remote",
            "Office",
            "Hybrid"
        ]
    )


    # Salary
    salary = st.number_input(
        "Salary Expectation (₹ per month)",
        min_value=10000,
        max_value=500000,
        value=50000,
        step=5000
    )


    # Location
    location = st.selectbox(
        "Location Preference",
        [
            "Ahmedabad",
            "Bangalore",
            "Mumbai",
            "Pune",
            "Delhi",
            "Hyderabad",
            "Chennai",
            "Vadodara",
            "Surat",
            "Remote"
        ]
    )


    # Work Life Balance
    work_balance = st.selectbox(
        "Work-Life Balance Preference",
        [
            "High",
            "Medium",
            "Low"
        ]
    )


# ============================================================
# FUNCTION:
# PREPARE INPUT USING SAME MLB AS TRAINING
# ============================================================

def prepare_input():

    # --------------------------------------------------------
    # Create raw input
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        {
            "Age": [age],

            "Education": [education],

            "Degree": [degree],

            "Skills": [
                ", ".join(skills)
            ],

            "Technical Skills": [
                ", ".join(technical_skills)
            ],

            "Soft Skills": [
                ", ".join(soft_skills)
            ],

            "Work Experience": [
                f"{work_experience} years"
            ],

            "Interests": [
                ", ".join(interests)
            ],

            "Career Interests": [
                career_interest
            ],

            "Personality/Strengths": [
                ", ".join(personality)
            ],

            "Preferred Work Environment": [
                work_environment
            ],

            "Salary Expectation": [
                salary
            ],

            "Location Preference": [
                location
            ],

            "Work-Life Balance Preference": [
                work_balance
            ]
        }
    )


    # --------------------------------------------------------
    # Multi-label columns
    # --------------------------------------------------------

    multi_label_columns = [
        "Skills",
        "Technical Skills",
        "Soft Skills",
        "Interests",
        "Personality/Strengths"
    ]


    # --------------------------------------------------------
    # Remove original multi-label columns
    # --------------------------------------------------------

    processed_input = input_data.drop(
        columns=multi_label_columns
    ).copy()


    # --------------------------------------------------------
    # Apply SAME trained MultiLabelBinarizer
    # --------------------------------------------------------

    for col in multi_label_columns:

        value = input_data[col].iloc[0]


        # Convert string to list
        if pd.isna(value):

            value_list = []

        elif str(value).strip() == "":

            value_list = []

        else:

            value_list = [
                item.strip()
                for item in str(value).split(",")
                if item.strip() != ""
            ]


        # Get trained MLB
        mlb = mlb_dict[col]


        # Transform using fitted MLB
        encoded = mlb.transform(
            [value_list]
        )


        # Create encoded DataFrame
        encoded_df = pd.DataFrame(
            encoded,
            columns=[
                f"{col}_{item}"
                for item in mlb.classes_
            ]
        )


        # Add encoded columns
        processed_input = pd.concat(
            [
                processed_input.reset_index(drop=True),

                encoded_df.reset_index(drop=True)
            ],
            axis=1
        )


    # --------------------------------------------------------
    # Convert Work Experience to numeric
    # --------------------------------------------------------

    processed_input[
        "Work Experience"
    ] = work_experience


    # --------------------------------------------------------
    # Get exact training columns
    # --------------------------------------------------------

    preprocessor = model.named_steps[
        "preprocessor"
    ]


    expected_columns = (
        preprocessor.feature_names_in_
    )


    # --------------------------------------------------------
    # Add missing columns
    # --------------------------------------------------------

    for col in expected_columns:

        if col not in processed_input.columns:

            processed_input[col] = 0


    # --------------------------------------------------------
    # Remove extra columns
    # --------------------------------------------------------

    processed_input = processed_input[
        expected_columns
    ]


    return processed_input


# ============================================================
# RECOMMEND BUTTON
# ============================================================

st.divider()


recommend_button = st.button(
    "🔮 Recommend Career",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if recommend_button:

    try:

        # Prepare input
        processed_input = prepare_input()


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            processed_input
        )


        recommended_career = (
            prediction[0]
        )


        # ----------------------------------------------------
        # Display main recommendation
        # ----------------------------------------------------

        st.success(
            f"🎯 Recommended Career: "
            f"{recommended_career}"
        )


        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = (
                model.predict_proba(
                    processed_input
                )[0]
            )


            classes = model.classes_


            probability_df = pd.DataFrame(
                {
                    "Career": classes,

                    "Probability": probabilities
                }
            )


            probability_df = (
                probability_df
                .sort_values(
                    "Probability",
                    ascending=False
                )
                .head(5)
                .reset_index(
                    drop=True
                )
            )


            probability_df[
                "Probability"
            ] = (
                probability_df[
                    "Probability"
                ] * 100
            ).round(2)


            # ------------------------------------------------
            # Top 5 recommendations
            # ------------------------------------------------

            st.subheader(
                "🏆 Top Career Recommendations"
            )


            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # Bar Chart
            # ------------------------------------------------

            chart_data = (
                probability_df
                .set_index("Career")
            )


            st.subheader(
                "📊 Career Recommendation Probability"
            )


            st.bar_chart(
                chart_data[
                    "Probability"
                ]
            )


        # ----------------------------------------------------
        # User Profile Summary
        # ----------------------------------------------------

        st.subheader(
            "👤 Your Profile Summary"
        )


        summary_col1, summary_col2 = (
            st.columns(2)
        )


        with summary_col1:

            st.write(
                f"**Age:** {age}"
            )

            st.write(
                f"**Education:** {education}"
            )

            st.write(
                f"**Degree:** {degree}"
            )

            st.write(
                f"**Work Experience:** "
                f"{work_experience} years"
            )

            st.write(
                f"**Career Interest:** "
                f"{career_interest}"
            )


        with summary_col2:

            st.write(
                f"**Work Environment:** "
                f"{work_environment}"
            )

            st.write(
                f"**Salary Expectation:** "
                f"₹{salary:,}"
            )

            st.write(
                f"**Location:** {location}"
            )

            st.write(
                f"**Work-Life Balance:** "
                f"{work_balance}"
            )


        # ----------------------------------------------------
        # Success message
        # ----------------------------------------------------

        st.info(
            f"""
            Based on the information provided, the model
            recommends **{recommended_career}** as the
            most suitable career option.
            """
        )


    except Exception as e:

        st.error(
            "Prediction Error"
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Women's Career Recommendation System | "
    "Machine Learning Project"
)