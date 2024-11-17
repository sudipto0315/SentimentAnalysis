import streamlit as st
import pandas as pd
from textblob import TextBlob
import cleantext
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Configure page
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add header with a catchy title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📊 Sentiment Analysis Dashboard</h1>", unsafe_allow_html=True)

st.sidebar.header("Navigation")
st.sidebar.markdown(
    """
    - 📄 **Analyze Text**
    - 📁 **Upload CSV**
    """
)

# Customize colors for Seaborn
sns.set_palette("Set2")

st.header('Sentiment Analysis')
with st.expander('📄 Analyze Text'):
    text = st.text_input('Enter text to analyze:', placeholder="Type your text here...")
    if text:
        blob = TextBlob(text)
        polarity = round(blob.sentiment.polarity, 2)
        subjectivity = round(blob.sentiment.subjectivity, 2)
        st.markdown(f"### Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Polarity", polarity)
        with col2:
            st.metric("Subjectivity", subjectivity)
        with col3:
            if polarity > 0:
                sentiment = 'Positive'
            elif polarity < 0:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'
            st.metric("Sentiment", sentiment)

    pre = st.text_input('Clean Text:', placeholder="Paste text to clean...")
    if pre:
        cleaned_text = cleantext.clean(
            pre, clean_all=False, extra_spaces=True, stopwords=True, lowercase=True, numbers=True, punct=True
        )
        st.markdown("### Cleaned Text")
        st.write(cleaned_text)

with st.expander('📁 Analyze CSV'):
    uploaded_file = st.file_uploader('Upload a CSV file for analysis')
    
    def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity
    
    def tanh(x):
        return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
    
    def analyze(x):
        tanh_score = tanh(x)
        if tanh_score >= 0.5:
            return 'Positive'
        elif tanh_score <= -0.5:
            return 'Negative'
        else:
            return 'Neutral'

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'Unnamed: 0' in df.columns:
            del df['Unnamed: 0']

        if 'text' in df.columns:
            df['text'] = df['text'].astype(str)  # Convert all values to strings
            df['text'] = df['text'].fillna('')   # Replace NaN values with empty strings
            df['score'] = df['text'].apply(score)
            df['analysis'] = df['score'].apply(analyze)
            # Display the first 10 rows of the dataframe
            st.markdown("### Sample Data")
            st.write(df.head(10))

            # Display summary metrics
            st.markdown("### Summary Metrics")
            total_positive = sum(df['analysis'] == 'Positive')
            total_negative = sum(df['analysis'] == 'Negative')
            total_neutral = sum(df['analysis'] == 'Neutral')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Positive", total_positive)
            with col2:
                st.metric("Negative", total_negative)
            with col3:
                st.metric("Neutral", total_neutral)

            # Data visualization
            st.markdown("### Data Visualizations")
            
            sentiment_counts = df['analysis'].value_counts()
            # Create two columns for side-by-side plots
            col1, col2 = st.columns(2)

            # Bar plot in the first column
            with col1:
                fig1, ax1 = plt.subplots()
                sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax1, palette="viridis")
                ax1.set_xlabel('Sentiment')
                ax1.set_ylabel('Count')
                ax1.set_title('Sentiment Analysis')
                st.pyplot(fig1)

            # Box plot in the second column
            with col2:
                fig2, ax2 = plt.subplots()
                sns.boxplot(data=df, x='analysis', y='score', ax=ax2, palette="coolwarm")
                ax2.set_title('Sentiment Score Distribution')
                ax2.set_xlabel('Sentiment')
                ax2.set_ylabel('Polarity Score')
                st.pyplot(fig2)

            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                trend_data = df.groupby(pd.Grouper(key='timestamp', freq='D'))['score'].mean().reset_index()
                fig3, ax3 = plt.subplots()
                sns.lineplot(data=trend_data, x='timestamp', y='score', ax=ax3)
                ax3.set_title('Sentiment Trends Over Time')
                ax3.set_xlabel('Date')
                ax3.set_ylabel('Average Sentiment Score')
                st.pyplot(fig3)
            else:
                st.warning("No 'timestamp' column found. Skipping time trend analysis.")

            @st.cache_data
            def convert_df(df):
                return df.to_csv().encode('utf-8')

            csv = convert_df(df)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name='sentiment_results.csv',
                mime='text/csv',
            )
        else:
            st.error("The uploaded CSV does not have a 'text' column.")