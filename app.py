# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="EduPro Learner Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# LOAD CUSTOM CSS
# ---------------------------------------------------

with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE SECTION
# ---------------------------------------------------

st.markdown("""
<div class="main-header">
    <h1>EduPro Learner Analytics Dashboard</h1>
    <p>
        Student Segmentation and Personalized Course Recommendation System
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# PROJECT DESCRIPTION
# ---------------------------------------------------

st.info("""
This dashboard uses uploaded EduPro learner transaction data for analytics,
behavior analysis, engagement tracking, revenue analysis, and learner insights.
The project is developed using Streamlit, Python, Pandas, Matplotlib, and Seaborn.
""")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv("data/EduPro Online Platform.csv")
    return data

df = load_data()

# ---------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------

df.drop_duplicates(inplace=True)

df.dropna(subset=['UserID', 'CourseID'], inplace=True)

df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])

df['Month'] = df['TransactionDate'].dt.month_name()

# ---------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------

total_courses = df.groupby('UserID')['CourseID'].count()

avg_spending = df.groupby('UserID')['Amount'].mean()

total_spending = df.groupby('UserID')['Amount'].sum()

preferred_category = df.groupby('UserID')['CourseCategory'] \
    .agg(lambda x: x.mode()[0])

preferred_level = df.groupby('UserID')['CourseLevel'] \
    .agg(lambda x: x.mode()[0])

avg_rating = df.groupby('UserID')['CourseRating'].mean()

diversity_score = df.groupby('UserID')['CourseCategory'].nunique()

enrollment_frequency = df.groupby('UserID')['CourseID'].count()

# ---------------------------------------------------
# LEARNER PROFILE
# ---------------------------------------------------

learner_profile = pd.DataFrame({
    'TotalCourses': total_courses,
    'AverageSpending': avg_spending,
    'TotalSpending': total_spending,
    'PreferredCategory': preferred_category,
    'PreferredLevel': preferred_level,
    'AverageRating': avg_rating,
    'DiversityScore': diversity_score,
    'EnrollmentFrequency': enrollment_frequency
})

learner_profile.reset_index(inplace=True)

# ---------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------

st.sidebar.markdown("## Dashboard Filters")

category_filter = st.sidebar.multiselect(
    "Select Course Category",
    options=df['CourseCategory'].unique(),
    default=df['CourseCategory'].unique()
)

level_filter = st.sidebar.multiselect(
    "Select Course Level",
    options=df['CourseLevel'].unique(),
    default=df['CourseLevel'].unique()
)

gender_filter = st.sidebar.multiselect(
    "Select Gender",
    options=df['UserGender'].unique(),
    default=df['UserGender'].unique()
)

min_spending = st.sidebar.slider(
    "Minimum Spending",
    int(df['Amount'].min()),
    int(df['Amount'].max()),
    int(df['Amount'].min())
)

filtered_df = df[
    (df['CourseCategory'].isin(category_filter)) &
    (df['CourseLevel'].isin(level_filter)) &
    (df['UserGender'].isin(gender_filter)) &
    (df['Amount'] >= min_spending)
]

# ---------------------------------------------------
# KPI SECTION
# ---------------------------------------------------

st.markdown("---")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        "Total Learners",
        filtered_df['UserID'].nunique()
    )
    st.caption("Represents the number of unique learners currently active in the filtered dataset.")

with kpi2:
    st.metric(
        "Total Courses",
        filtered_df['CourseID'].nunique()
    )
    st.caption("Shows the number of unique courses available under selected filters.")

with kpi3:
    st.metric(
        "Total Revenue",
        f"₹ {round(filtered_df['Amount'].sum(), 2)}"
    )
    st.caption("Indicates total learner spending generated from course enrollments.")

with kpi4:
    st.metric(
        "Average Rating",
        round(filtered_df['CourseRating'].mean(), 2)
    )
    st.caption("Displays average learner satisfaction based on course ratings.")

# ---------------------------------------------------
# TABS
# ---------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Learner Insights",
    "Course Analytics",
    "Revenue Analysis",
    "Recommendations"
])

# ---------------------------------------------------
# OVERVIEW TAB
# ---------------------------------------------------

with tab1:

    st.subheader("Platform Overview")

    st.markdown("""
This section provides a high-level understanding of learner participation,
course preferences, and engagement patterns across the EduPro platform.
""")

    graph1, graph2 = st.columns(2)

    with graph1:

        fig, ax = plt.subplots(figsize=(8, 5))

        filtered_df['CourseCategory'].value_counts().plot(
            kind='bar',
            ax=ax
        )

        plt.title("Course Category Distribution")
        plt.xlabel("Course Category")
        plt.ylabel("Enrollments")
        plt.xticks(rotation=45)

        st.pyplot(fig)

        st.success("""
Insight:
Technology-oriented domains such as Data Science, Cloud Computing,
and AI attract the highest learner enrollments.
""")

    with graph2:

        fig, ax = plt.subplots(figsize=(7, 5))

        filtered_df['CourseLevel'].value_counts().plot(
            kind='pie',
            autopct='%1.1f%%',
            ax=ax
        )

        ax.set_ylabel("")

        plt.title("Course Level Distribution")

        st.pyplot(fig)

        st.success("""
Insight:
Beginner and Intermediate courses dominate enrollments,
indicating strong participation from early-stage learners.
""")

# ---------------------------------------------------
# LEARNER INSIGHTS TAB
# ---------------------------------------------------

with tab2:

    st.subheader("Learner Behavior Analytics")

    st.markdown("""
This section analyzes learner engagement, course exploration behavior,
and participation intensity across the platform.
""")

    graph1, graph2 = st.columns(2)

    with graph1:

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.histplot(
            learner_profile['TotalCourses'],
            bins=10,
            ax=ax
        )

        plt.title("Total Courses Enrolled")
        plt.xlabel("Courses")
        plt.ylabel("Learners")

        st.pyplot(fig)

        st.success("""
Insight:
A smaller segment of highly active learners contributes
significantly to overall platform engagement.
""")

    with graph2:

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.histplot(
            learner_profile['DiversityScore'],
            bins=10,
            ax=ax
        )

        plt.title("Learner Diversity Score")
        plt.xlabel("Diversity Score")
        plt.ylabel("Learners")

        st.pyplot(fig)

        st.success("""
Insight:
Many learners explore multiple course domains,
indicating interest in interdisciplinary learning.
""")

    # Drill-down Filter

    st.markdown("### Drill-Down Analysis")

    selected_category = st.selectbox(
        "Select Category for Detailed Analysis",
        filtered_df['CourseCategory'].unique()
    )

    category_data = filtered_df[
        filtered_df['CourseCategory'] == selected_category
    ]

    st.dataframe(category_data.head(10))

# ---------------------------------------------------
# COURSE ANALYTICS TAB
# ---------------------------------------------------

with tab3:

    st.subheader("Course Analytics")

    st.markdown("""
This section focuses on course ratings, learner demographics,
and category-level engagement patterns.
""")

    graph1, graph2 = st.columns(2)

    with graph1:

        category_ratings = filtered_df.groupby(
            'CourseCategory'
        )['CourseRating'].mean().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))

        category_ratings.plot(
            kind='bar',
            ax=ax
        )

        plt.title("Average Ratings by Category")
        plt.xlabel("Course Category")
        plt.ylabel("Average Rating")
        plt.xticks(rotation=45)

        st.pyplot(fig)

        st.success("""
Insight:
Highly rated categories demonstrate stronger learner
satisfaction and engagement levels.
""")

    with graph2:

        fig, ax = plt.subplots(figsize=(7, 5))

        filtered_df['UserGender'].value_counts().plot(
            kind='pie',
            autopct='%1.1f%%',
            ax=ax
        )

        ax.set_ylabel("")

        plt.title("Gender Distribution")

        st.pyplot(fig)

        st.success("""
Insight:
The platform demonstrates balanced participation
across learner demographics.
""")

# ---------------------------------------------------
# REVENUE ANALYSIS TAB
# ---------------------------------------------------

with tab4:

    st.subheader("Revenue and Enrollment Trends")

    st.markdown("""
This section analyzes learner spending behavior,
revenue distribution, and enrollment trends over time.
""")

    graph1, graph2 = st.columns(2)

    with graph1:

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.histplot(
            filtered_df['Amount'],
            bins=10,
            ax=ax
        )

        plt.title("Revenue Distribution")
        plt.xlabel("Amount")
        plt.ylabel("Frequency")

        st.pyplot(fig)

        st.success("""
Insight:
Most learners fall into moderate spending ranges,
while premium learners contribute significantly higher revenue.
""")

    with graph2:

        monthly_enrollments = filtered_df.groupby(
            'Month'
        )['CourseID'].count()

        fig, ax = plt.subplots(figsize=(8, 5))

        monthly_enrollments.plot(
            kind='line',
            marker='o',
            ax=ax
        )

        plt.title("Monthly Enrollment Trend")
        plt.xlabel("Month")
        plt.ylabel("Enrollments")

        st.pyplot(fig)

        st.success("""
Predictive Insight:
Current trends suggest technology-oriented courses
will continue driving higher enrollments in upcoming periods.
""")

# ---------------------------------------------------
# RECOMMENDATIONS TAB
# ---------------------------------------------------

with tab5:

    st.subheader("""Business Recommendations""")
    st.markdown("""
#### 1. Focus on Beginner Learners
Expand beginner-friendly learning pathways since beginner courses dominate enrollments.

#### 2. Promote High-Rated Categories
Invest more in highly rated domains such as AI, Data Science, and Cloud Computing.

#### 3. Increase Learner Retention
Target low-engagement learners through personalized campaigns and engagement initiatives.

#### 4. Expand Premium Offerings
Introduce advanced certifications and premium programs for high-spending learners.

#### 5. Cross-Domain Learning
Encourage learners to explore multiple domains through bundled learning tracks.

#### 6. Seasonal Enrollment Planning
Use monthly enrollment patterns to optimize marketing and promotional campaigns.
""")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.markdown("""
<div class="footer">
EduPro Learner Analytics Dashboard
</div>
""", unsafe_allow_html=True)