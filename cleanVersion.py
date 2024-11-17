from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext
import matplotlib.pyplot as plt
import seaborn as sns
import math

st.header('Sentiment Analysis')
with st.expander('Analyze Text'):
    text=st.text_input('Text here: ')
    if text:
        blob=TextBlob(text)
        polarity=round(blob.sentiment.polarity,2)
        subjectivity=round(blob.sentiment.subjectivity,2)
        st.write('Polarity: ',polarity)
        st.write('Subjectivity: ',subjectivity)
        if polarity>0:
            sentiment='Positive'
        elif polarity<0:
            sentiment='Negative'
        else:
            sentiment='Neutral'
        st.write('Sentiment: ',sentiment)
    pre=st.text_input('Clean Text: ')
    if pre:
        st.write(cleantext.clean(pre, clean_all= False, extra_spaces=True ,
                                 stopwords=True ,lowercase=True ,numbers=True , punct=True))

with st.expander('Analyze CSV'):
    uploaded_file=st.file_uploader('Upload file')
    def score(x):
        blob1=TextBlob(x)
        return blob1.sentiment.polarity
    
    def tanh(x):
        return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
    
    def analyze(x):
        tanh_score=tanh(x)
        if tanh_score>=0.5:
            return 'Positive'
        elif tanh_score<=-0.5:
            return 'Negative'
        else:
            return 'Neutral'
        
    if uploaded_file:
        df=pd.read_csv(uploaded_file)
        if 'Unnamed: 0' in df.columns:
            del df['Unnamed: 0']

        if 'text' in df.columns:
            df['text']=df['text'].astype(str)  # Convert all values to strings
            df['text']=df['text'].fillna('')   # Replace NaN values with empty strings
            df['score']=df['text'].apply(score)
            df['analysis']=df['score'].apply(analyze)
            st.write(df.head(10))
            @st.cache_data
            def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv=convert_df(df)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='sentiment.csv',
                mime='text/csv',
            )

            # Analysis of total positive, neutral, and negative sentiments
            total_positive = sum(df['analysis'] == 'Positive')
            total_negative = sum(df['analysis'] == 'Negative')
            total_neutral = sum(df['analysis'] == 'Neutral')
            
            st.write(f"Total Positive: {total_positive}")
            st.write(f"Total Negative: {total_negative}")
            st.write(f"Total Neutral: {total_neutral}")

            # Plotting the analysis

            # Count Plot(Bar Plot)
            sentiment_counts = df['analysis'].value_counts()
            fig1, ax = plt.subplots()
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax)
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Count')
            ax.set_title('Sentiment Analysis')
            st.pyplot(fig1)

            # Distribution of Sentiment Scores (Box Plot)
            fig2, ax3 = plt.subplots()
            sns.boxplot(data=df, x='analysis', y='score', ax=ax3, palette='coolwarm')
            ax3.set_title('Sentiment Score Distribution')
            ax3.set_xlabel('Sentiment')
            ax3.set_ylabel('Polarity Score')
            st.pyplot(fig2)

            # Time Trend Line Plot (if applicable)
            if 'timestamp' in df.columns:  # Assuming a 'timestamp' column exists
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                trend_data = df.groupby(pd.Grouper(key='timestamp', freq='D'))['score'].mean().reset_index()
                fig3, ax4 = plt.subplots()
                sns.lineplot(data=trend_data, x='timestamp', y='score', ax=ax4)
                ax4.set_title('Sentiment Trends Over Time')
                ax4.set_xlabel('Date')
                ax4.set_ylabel('Average Sentiment Score')
                st.pyplot(fig3)
            else:
                st.warning("No 'timestamp' column found. Skipping time trend analysis.")
        else:
            st.error("The uploaded CSV file does not contain a 'text' column.")