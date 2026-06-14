import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Loading CSS
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Title
st.markdown("""
<div class="main-header">
    <p>
        Student Segmentation and Personalized Course Recommendation System for EduPro
    </p>
</div>
""", unsafe_allow_html=True)

# Project Description
st.info("""
This dashboard uses uploaded EduPro learner transaction data for analytics,
behavior analysis, engagement tracking, revenue analysis, and learner insights.
The project is developed using Streamlit, Python, Pandas, Matplotlib, and Seaborn.
""")

# Data Loading
@st.cache_data
def load_data():
    data = pd.read_csv("data/EduPro Online Platform.csv")
    return data

df = load_data()

# Data Cleaning
df.drop_duplicates(inplace=True)

df.dropna(subset=['UserID', 'CourseID'], inplace=True)

df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])

df['Month'] = df['TransactionDate'].dt.month_name()

# Feature Engineering
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

# Learner Profile
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

# Filters
st.sidebar.markdown(
    "<h2 class='sidebar-title'>🔍 Filters</h2>",
    unsafe_allow_html=True
)

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

# KPIs
st.markdown("---")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        "Total Learners",
        filtered_df['UserID'].nunique()
    )
    st.markdown("""
        <div class="kpi-description">
        Represents the number of unique learners currently active in the filtered dataset.
        </div>
        """, unsafe_allow_html=True)
    
with kpi2:
    st.metric(
        "Total Courses",
        filtered_df['CourseID'].nunique()
    )
    st.markdown("""
        <div class="kpi-description">
        Shows the number of unique courses available under selected filters.
        </div>
        """, unsafe_allow_html=True)

with kpi3:
    st.metric(
        "Total Revenue",
        f"₹ {round(filtered_df['Amount'].sum(), 2)}"
    )
    st.markdown("""
        <div class="kpi-description">
        Indicates total learner spending generated from course enrollments.
        </div>
        """, unsafe_allow_html=True)

with kpi4:
    st.metric(
        "Average Rating",
        round(filtered_df['CourseRating'].mean(), 2)
    )
    st.markdown("""
        <div class="kpi-description">
        Displays average learner satisfaction based on course ratings.
        </div>
        """, unsafe_allow_html=True)

# Key Fidings 
st.markdown("---")

st.markdown("""
<div class="section-title">
Key Findings
</div>
""", unsafe_allow_html=True)

col1,col2,col3=st.columns(3)

with col1:
    st.markdown(f"""
    <div class="finding-card">
    <h4>Most Popular Category</h4>
    <p>{filtered_df['CourseCategory'].mode()[0]}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="finding-card">
    <h4>Highest Rated Category</h4>
    <p>{filtered_df.groupby('CourseCategory')['CourseRating'].mean().idxmax()}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="finding-card">
    <h4>Average Learner Spend</h4>
    <p>₹ {round(filtered_df['Amount'].mean(),2)}</p>
    </div>
    """, unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Learner Insights",
    "Course Analytics",
    "Revenue Analysis",
    "Recommendations"
])

# Overview Tab
with tab1:

    st.markdown("""
        <div class="tab-title">
        Platform Overview
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""
        <div class="tab-description">
        This section provides a high-level understanding of learner participation,
        course preferences, and engagement patterns across the EduPro platform.
        </div>
        """, unsafe_allow_html=True)
   
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

        st.markdown("""
        <div class="insight-box">
        <b>Insight:</b><br>
        Technology-oriented domains such as Data Science, Cloud Computing, and AI attract the highest learner enrollments.
        </div>
        """, unsafe_allow_html=True)

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

        st.markdown("""
        <div class="insight-box">
        <b>Insight:</b><br>
        Beginner and Intermediate courses dominate enrollments, indicating strong participation from early-stage learners.
        </div>
        """, unsafe_allow_html=True)

# Learner Insights Tab
with tab2:

    st.markdown("""
        <div class="tab-title">
        Learner Behavior Analytics
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""
        <div class="tab-description">
        This section analyzes learner engagement, course exploration behavior,
        and participation intensity across the platform.
        </div>
        """, unsafe_allow_html=True)
   
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

        st.markdown("""
        <div class="insight-box">
        <b>Insight:</b><br>
        A smaller segment of highly active learners contributes significantly to overall platform engagement.
        </div>
        """, unsafe_allow_html=True)

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

        st.markdown("""
        <div class="insight-box">
        <b>Insight:</b><br>
        Many learners explore multiple domains, indicating strong interdisciplinary learning behavior.
        </div>
        """, unsafe_allow_html=True)

# Course Analytics Tab
with tab3:

    st.markdown("""
        <div class="tab-title">
        Course Analytics
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""
        <div class="tab-description">
        This section focuses on course ratings, learner demographics,
        and category-level engagement patterns.
        </div>
        """, unsafe_allow_html=True)
    
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

        st.markdown("""
        <div class="insight-box">
        <b>Insight:</b><br>
        Highly rated categories demonstrate stronger learner satisfaction and engagement levels.
        </div>
        """, unsafe_allow_html=True)

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

        st.markdown("""
        <div class="insight-box">
        <b>Insight:</b><br>
        The platform demonstrates balanced participation across learner demographics.
        </div>
        """, unsafe_allow_html=True)

# Revenue Analysis Tab
with tab4:

    st.markdown("""
        <div class="tab-title">
        Revenue and Enrollment Trends
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""
        <div class="tab-description">
        This section analyzes learner spending behavior, revenue distribution, and enrollment trends over time.
        </div>
        """, unsafe_allow_html=True)
   
    rev1,rev2,rev3 = st.columns(3)

    with rev1:
        st.metric(
        "Average Transaction",
        f"₹ {round(filtered_df['Amount'].mean(),2)}"
    )

    with rev2:
        st.metric(
        "Highest Revenue Category",
        filtered_df.groupby(
            'CourseCategory'
        )['Amount'].sum().idxmax()
    )

    with rev3:
        st.metric(
        "Maximum Transaction",
        f"₹ {round(filtered_df['Amount'].max(),2)}"
    )
        
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

        st.markdown("""
        <div class="insight-box">
        <b>Insight:</b><br>
        Insight:
        Most learners fall into moderate spending ranges, while premium learners contribute significantly higher revenue.
        </div>
        """, unsafe_allow_html=True)

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

        st.markdown("""
        <div class="forecast-box">
        <h4>Trend-Based Forecast</h4>
        <p>Current enrollment patterns suggest that technology-focused categories are likely 
                    to continue attracting higher learner participation.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
        <b>Insight:</b><br>
        Predictive Insight:
        Current trends suggest technology-oriented courses will continue driving higher enrollments in upcoming periods.
        </div>
        """, unsafe_allow_html=True)

# Recomendations Tab
with tab5:

    st.markdown("""
        <div class="tab-title">
        Business Recommendations
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="recommendation-card">
    <h4>Focus on Beginner Learners</h4>
    <p>Expand beginner learning pathways because they contribute the largest share of enrollments.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="recommendation-card">
    <h4>Promote High-Rated Categories</h4>
    <p>Invest more in highly rated domains such as AI, Data Science, and Cloud Computing.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="recommendation-card">
    <h4>Increase Learner Retention</h4>
    <p>Target low-engagement learners through personalized campaigns and engagement initiatives.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="recommendation-card">
    <h4>Expand Premium Offerings</h4>
    <p>Introduce advanced certifications and premium programs for high-spending learners.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="recommendation-card">
    <h4>Cross-Domain Learning</h4>
    <p>Encourage learners to explore multiple domains through bundled learning tracks.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="recommendation-card">
    <h4>Seasonal Enrollment Planning</h4>
    <p>Use monthly enrollment patterns to optimize marketing and promotional campaigns.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")

st.markdown("""
<div class="footer">
EduPro Learner Analytics Dashboard
</div>
""", unsafe_allow_html=True)