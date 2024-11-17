# Sentiment Analysis Dashboard

## Overview

The Sentiment Analysis Dashboard is a web application built with Streamlit that allows users to analyze the sentiment of text data. Users can input text directly or upload a CSV file containing text data for analysis. The application provides sentiment scores, visualizations, and the ability to download the results.

## Features

- **Analyze Text**: Input text directly to get sentiment polarity, subjectivity, and overall sentiment (Positive, Neutral, Negative).
- **Clean Text**: Clean text by removing extra spaces, stopwords, converting to lowercase, and removing numbers and punctuation.
- **Analyze CSV**: Upload a CSV file containing text data for batch sentiment analysis.
- **Visualizations**: Generate bar plots and box plots to visualize sentiment distribution and trends over time.
- **Download Results**: Download the analyzed results as a CSV file.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/sudipto0315/SentimentAnalysis.git
   cd SentimentAnalysis
   ```
2. **Create a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```
2. **Open your web browser** and navigate to `http://localhost:8501` to access the Sentiment Analysis Dashboard.

## How It Works

### Analyze Text

1. **Enter text** in the "Analyze Text" section.
2. The application uses `TextBlob` to calculate the polarity and subjectivity of the text.
3. The sentiment (Positive, Neutral, Negative) is determined based on the polarity score.
4. Results are displayed with metrics for polarity, subjectivity, and sentiment.

### Clean Text

1. **Enter text** in the "Clean Text" section.
2. The application uses `clean-text` to clean the text by removing extra spaces, stopwords, converting to lowercase, and removing numbers and punctuation.
3. The cleaned text is displayed.

### Analyze CSV

1. **Upload a CSV file** containing a column named `text`.
2. The application reads the CSV file and processes each text entry to calculate sentiment scores.
3. Sentiment analysis results are displayed in a table.
4. Summary metrics for total positive, neutral, and negative sentiments are shown.
5. Visualizations include:
   - Bar plot of sentiment counts.
   - Box plot of sentiment score distribution.
   - Line plot of sentiment trends over time (if a `timestamp` column is present).
6. The analyzed results can be downloaded as a CSV file.

## Example

### Sample Data

```csv
text,timestamp
"I love this product!",2024-11-01
"This is the worst service ever.",2024-11-02
"Not bad, could be better.",2024-11-03
```
