# streamlit_dashboard.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import pipeline

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('Tweets.csv')
    df = df[["airline", "text", "airline_sentiment", "tweet_created"]]
    df = df.rename(columns={
        "airline": "Brand",
        "text": "Tweet",
        "airline_sentiment": "Sentiment",
        "tweet_created": "Date"
    })
    df["Date"] = pd.to_datetime(df["Date"])
    df["DateOnly"] = df["Date"].dt.date
    return df

df = load_data()

# Sidebar filters
st.sidebar.title("Filters")
brands = st.sidebar.multiselect("Select Brands", df["Brand"].unique(), default=df["Brand"].unique())
sentiments = st.sidebar.multiselect("Select Sentiments", df["Sentiment"].unique(), default=df["Sentiment"].unique())
date_range = st.sidebar.date_input("Date Range", [df["DateOnly"].min(), df["DateOnly"].max()])

# Filter data
filtered_df = df[
    (df["Brand"].isin(brands)) &
    (df["Sentiment"].isin(sentiments)) &
    (df["DateOnly"] >= date_range[0]) &
    (df["DateOnly"] <= date_range[1])
]

# Dashboard
st.title(" Brand Sentiment Monitor Dashboard")

# Show sentiment distribution
st.subheader("Sentiment Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(data=filtered_df, x="Sentiment", palette="Set2", ax=ax1)
st.pyplot(fig1)

# Sentiment trend
st.subheader("Sentiment Trend Over Time")

trend = filtered_df.groupby(["DateOnly", "Brand", "Sentiment"]).size().unstack().fillna(0)

for brand in filtered_df["Brand"].unique():
    if brand in trend.index.get_level_values(1):
        brand_data = trend.xs(brand, level=1)
        st.markdown(f" Trend for {brand}")
        fig2, ax2 = plt.subplots()
        brand_data.plot(ax=ax2)
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Tweet Count")
        ax2.grid(True)
        st.pyplot(fig2)

# Named Entity Recognition
st.subheader(" Named Entity Recognition (NER)")
tweet_example = st.selectbox("Select a Tweet", filtered_df["Tweet"].sample(5).values)
if st.button("Run NER"):
    ner_pipe = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    ner_results = ner_pipe(tweet_example)
    for entity in ner_results:
        st.write(f"**{entity['word']}** ({entity['entity_group']}) - Score: {round(entity['score'], 3)}")

# Footer
st.markdown("---")
st.markdown("Built by Huda Mawood – Brand Sentiment Monitor ")

