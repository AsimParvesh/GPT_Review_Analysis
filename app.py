import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import tensorflow as tf
import pickle
from wordcloud import WordCloud
from collections import Counter
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Bidirectional, LSTM

# NLTK Setup
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


import nltk

for resource in ['punkt', 'stopwords', 'wordnet']:
    try:
        if resource == 'punkt':
            nltk.data.find('tokenizers/punkt')
        else:
            nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)


# Load Model and Tools
model = tf.keras.models.load_model(
    'models/sentiment_model.h5',
    custom_objects={'Bidirectional': Bidirectional, 'LSTM': LSTM}
)
with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('models/label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Text Preprocessing Functions
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

from nltk.tokenize import word_tokenize

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    try:
        tokens = word_tokenize(text)
    except LookupError:
        nltk.download('punkt')
        tokens = word_tokenize(text)

    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)


def predict_sentiment(review_text):
    cleaned = clean_text(review_text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=150)
    prediction = model.predict(padded)
    predicted_class = prediction.argmax(axis=-1)[0]
    sentiment = encoder.inverse_transform([predicted_class])[0]
    return sentiment

# Load Data
df = pd.read_csv("data/cleaned_reviews.csv")

# Preprocess for Insights
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['month'] = df['date'].dt.to_period('M').astype(str)
df['review_length'] = df['review'].apply(len)
df['sentiment_score'] = df['rating'].apply(lambda x: 1 if x >= 4 else (0 if x == 3 else -1))
df['sentiment'] = df['rating'].apply(lambda x: 'positive' if x >= 4 else ('neutral' if x == 3 else 'negative'))

# Sidebar
st.set_page_config(page_title="AI Echo: Sentiment Insights & Prediction", layout="wide")
st.sidebar.title("🧠 AI Echo")
st.sidebar.markdown("Navigate between **Sentiment Prediction** and **Review Insights**.")
page = st.sidebar.radio("Choose Section", ["🔍 Sentiment Prediction", "📊 Review Insights"])

# ----------------------------------------
# SENTIMENT PREDICTION TAB
# ----------------------------------------
if page == "🔍 Sentiment Prediction":
    st.title("🔍 Predict Review Sentiment")
    st.markdown("Enter a review below to classify its sentiment.")

    review_input = st.text_area("✍️ Review Text", height=200, placeholder="Type your review here...")

    if st.button("Predict"):
        if review_input.strip():
            sentiment = predict_sentiment(review_input)
            st.success(f"**Predicted Sentiment:** {sentiment}")
        else:
            st.warning("Please enter a review before predicting.")

# ----------------------------------------
# REVIEW INSIGHTS TAB
# ----------------------------------------
elif page == "📊 Review Insights":
    st.title("📊 User Review Insights")

    tab1, tab2 = st.tabs(["📈 General Insights", "📅 Sentiment Analysis"])

    # ------- TAB 1 -------
    with tab1:
            # Insight 1
            st.subheader("1. Distribution of Review Ratings")
            fig1, ax1 = plt.subplots()
            sns.countplot(x='rating', data=df, palette='coolwarm', ax=ax1)
            rating_counts = df['rating'].value_counts().sort_index()
            for i, count in enumerate(rating_counts):
                ax1.text(i, count + 5, str(count), ha='center')
            ax1.set_title('Distribution of Review Ratings')
            st.pyplot(fig1)

            # Insight 2
            st.subheader("2. Helpful vs. Not Helpful Reviews")
            fig2, ax2 = plt.subplots()
            df['helpful_flag'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'], startangle=90, ax=ax2)
            ax2.set_ylabel('')
            ax2.set_title('Helpful vs. Not Helpful Reviews')
            st.pyplot(fig2)

            # Insight 3
            st.subheader("3. Common Keywords in Positive vs. Negative Reviews")
            def plot_wordcloud(text, title):
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(text))
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title(title, fontsize=16)
                ax.axis('off')
                st.pyplot(fig)

            plot_wordcloud(df[df['sentiment'] == 'positive']['review'], '☁️ Word Cloud - Positive Reviews')
            plot_wordcloud(df[df['sentiment'] == 'negative']['review'], '☁️ Word Cloud - Negative Reviews')

            # Insight 4
            st.subheader("4. Average Rating Over Time")
            monthly_avg = df.groupby('month')['rating'].mean().reset_index()
            fig4, ax4 = plt.subplots(figsize=(12, 6))
            sns.lineplot(data=monthly_avg, x='month', y='rating', marker='o', linewidth=2, ax=ax4)
            ax4.set_title("📆 Average Rating Over Time")
            plt.xticks(rotation=45)
            st.pyplot(fig4)

            # Insight 5
            st.subheader("5. Ratings by User Location")
            df_location = df.dropna(subset=['location', 'rating'])
            location_avg = df_location.groupby('location')['rating'].mean().reset_index()
            fig5 = px.choropleth(location_avg, locations='location', locationmode='country names', color='rating',
                                color_continuous_scale='Plasma', title='🌍 Average Rating by Country')
            st.plotly_chart(fig5)


            # Insight 6
            st.subheader("6. Web vs Mobile: Which Gets Better Reviews?")
            platform_avg = df.groupby('platform')['rating'].mean().reset_index()
            fig6, ax6 = plt.subplots()
            sns.barplot(data=platform_avg, x='platform', y='rating', palette='pastel', ax=ax6)
            ax6.set_ylim(1, 5)
            ax6.set_title("💻📱 Average Rating by Platform")

            # Add value labels on top of each bar
            for p in ax6.patches:
                height = p.get_height()
                ax6.text(p.get_x() + p.get_width() / 2., height + 0.05, f'{height:.2f}', ha='center', va='bottom', fontsize=10)

            st.pyplot(fig6)


            # Insight 7
            st.subheader("7. Verified vs Non-Verified Users")
            verified_avg = df.groupby('verified_purchase')['rating'].mean().reset_index()
            fig7, ax7 = plt.subplots()
            sns.barplot(data=verified_avg, x='verified_purchase', y='rating', palette='Set2', ax=ax7)
            ax7.set_ylim(1, 5)
            ax7.set_title("✅❌ Average Rating by Verified Purchase")

            # Add value labels on top of each bar
            for p in ax7.patches:
                height = p.get_height()
                ax7.text(p.get_x() + p.get_width() / 2., height + 0.05, f'{height:.2f}', ha='center', va='bottom', fontsize=10)

            st.pyplot(fig7)


            # Insight 8
            st.subheader("8. Average Review Length by Rating")
            length_by_rating = df.groupby('rating')['review_length'].mean().reset_index()
            fig8, ax8 = plt.subplots()
            sns.barplot(data=length_by_rating, x='rating', y='review_length', palette='coolwarm', ax=ax8)
            for p in ax8.patches:
                ax8.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=10)
            ax8.set_title('✍️📏 Avg Review Length by Rating')
            st.pyplot(fig8)

            # Insight 9
            st.subheader("9. Top Words in 1-Star Reviews")
            one_star_reviews = df[df['rating'] == 1]['review']
            stop_words = set(stopwords.words('english'))

            def clean_text(text):
                text = re.sub(r'\W', ' ', text).lower()
                return ' '.join([word for word in text.split() if word not in stop_words])

            cleaned_reviews = one_star_reviews.apply(clean_text)
            word_counts = Counter(" ".join(cleaned_reviews).split())
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
            fig9, ax9 = plt.subplots(figsize=(10, 6))
            ax9.imshow(wordcloud, interpolation='bilinear')
            ax9.set_title("💬 Common Words in 1-Star Reviews")
            ax9.axis('off')
            st.pyplot(fig9)


            # Insight 10
            st.subheader("10. Average Rating by ChatGPT Version")
            version_avg_rating = df.groupby('version')['rating'].mean().sort_values(ascending=False)
            fig10, ax10 = plt.subplots(figsize=(10, 6))
            bars = version_avg_rating.plot(kind='bar', color='lightblue', ax=ax10)
            # Set title and y-axis limit
            ax10.set_title("📱🧪 Average Rating by ChatGPT Version")
            ax10.set_ylim(1, 5)
            # Add value labels on top of each bar
            for p in ax10.patches:
                height = p.get_height()
                ax10.text(p.get_x() + p.get_width() / 2., height + 0.05, f'{height:.2f}', ha='center', va='bottom', fontsize=10)
            st.pyplot(fig10)



    # ------- TAB 2 -------
    with tab2:
            
            st.header("1. What is the overall sentiment of user reviews?")
            sentiment_distribution = df['sentiment'].value_counts(normalize=True)
            fig, ax = plt.subplots()
            bars = sentiment_distribution.plot(kind='bar', color='lightgreen', ax=ax)
            ax.set_title("Sentiment Distribution of Reviews")
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Proportion")
            # Add value labels inside the bars
            for p in ax.patches:
                height = p.get_height()
                ax.text(
                    p.get_x() + p.get_width() / 2.,
                    height - 0.05,  # Position inside the bar
                    f'{height:.2f}',
                    ha='center',
                    va='top',
                    fontsize=10,
                    color='black'  # Use 'white' if bars are dark
                )
            st.pyplot(fig)



            st.header("2. How does sentiment vary by rating?")
            rating_sentiment = df.groupby(['rating', 'sentiment']).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = rating_sentiment.plot(kind='bar', stacked=True, ax=ax, colormap="coolwarm_r")
            ax.set_title("Sentiment Distribution by Rating")
            ax.set_xlabel("Rating")
            ax.set_ylabel("Number of Reviews")
            # Add value labels on stacked bars
            for container in ax.containers:
                for bar in container:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2,
                                f'{int(height)}', ha='center', va='center', fontsize=9, color='black')
            st.pyplot(fig)



            st.header("3. Which keywords or phrases are most associated with each sentiment class?")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Positive")
                pos_wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df[df['sentiment'] == 'positive']['review']))
                st.image(pos_wc.to_array(), use_container_width=True)
            with col2:
                st.subheader("Negative")
                neg_wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df[df['sentiment'] == 'negative']['review']))
                st.image(neg_wc.to_array(), use_container_width=True)
            with col3:
                st.subheader("Neutral")
                neu_wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df[df['sentiment'] == 'neutral']['review']))
                st.image(neu_wc.to_array(), use_container_width=True)

            st.header("4. How has sentiment changed over time?")
            monthly_sentiment = df.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(12, 6))
            monthly_sentiment.plot(kind='line', marker='o', ax=ax, colormap='coolwarm_r')
            ax.set_title("Sentiment Change Over Time")
            ax.set_xlabel("Month")
            ax.set_ylabel("Number of Reviews")
            st.pyplot(fig)


            st.header("5. Do verified users tend to leave more positive or negative reviews?")
            verified_sentiment = df.groupby(['verified_purchase', 'sentiment']).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = verified_sentiment.plot(kind='bar', ax=ax, colormap='RdYlGn')
            ax.set_title("Sentiment by Verified Purchase")
            ax.set_xlabel("Verified Purchase")
            ax.set_ylabel("Count")
            # Add value labels on bars
            for container in bars.containers:
                for bar in container:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            height / 2,
                            f'{int(height)}',
                            ha='center',
                            va='center',
                            fontsize=10,
                            color='black'
                        )
            st.pyplot(fig)


            st.header("6. Are longer reviews more likely to be negative or positive?")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='sentiment', y='review_length', data=df, palette='Set2', ax=ax)
            ax.set_title("Review Length by Sentiment")
            st.pyplot(fig)

            avg_lengths = df.groupby('sentiment')['review_length'].mean()
            st.subheader("Average Review Length by Sentiment")
            st.dataframe(avg_lengths.reset_index().rename(columns={'review_length': 'Average Length'}))


            st.header("7. Which locations show the most positive or negative sentiment?")

            # Group sentiment by location
            location_sentiment = df.groupby(['location', 'sentiment']).size().unstack(fill_value=0)

            # Plot the bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            location_sentiment.plot(kind='bar', ax=ax, colormap='coolwarm_r')

            # Add value labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%d', label_type='edge', fontsize=8)

            # Chart formatting
            ax.set_title("Sentiment by Location")
            ax.set_xlabel("Location")
            ax.set_ylabel("Number of Reviews")
            ax.tick_params(axis='x', rotation=45)

            st.pyplot(fig)

            if 'positive' in location_sentiment:
                top_pos_loc = location_sentiment['positive'].idxmax()
                top_neg_loc = location_sentiment['negative'].idxmax()
                st.write(f"✅ Most positive: **{top_pos_loc}**")
                st.write(f"📉 Most negative: **{top_neg_loc}**")



            st.header("8. Is there a difference in sentiment across platforms?")

            # Ensure 'month' is in datetime format and sorted
            df['month'] = pd.to_datetime(df['month'])
            df['month'] = df['month'].dt.to_period('M').astype(str)  # Convert to 'YYYY-MM'

            # Group by month, platform, and sentiment
            monthly_platform_sentiment = df.groupby(['month', 'platform', 'sentiment']).size().unstack(fill_value=0).reset_index()

            # Plot 3 separate line charts: Positive, Neutral, Negative
            sentiments = ['positive', 'neutral', 'negative']
            for sentiment in sentiments:
                st.subheader(f"{sentiment.capitalize()} Sentiment Over Time: Web vs Mobile")

                # Pivot data for the sentiment to compare platforms
                sentiment_data = monthly_platform_sentiment.pivot(index='month', columns='platform', values=sentiment).fillna(0)

                # Line plot
                fig, ax = plt.subplots(figsize=(10, 5))
                sentiment_data.plot(kind='line', marker='o', ax=ax)

                ax.set_title(f"{sentiment.capitalize()} Sentiment Trend: Mobile vs Web")
                ax.set_xlabel("Month")
                ax.set_ylabel("Number of Reviews")
                ax.legend(title="Platform")
                ax.grid(True)

                # Add value labels
                for line in ax.lines:
                    for x, y in zip(line.get_xdata(), line.get_ydata()):
                        ax.text(x, y + 1, f'{int(y)}', ha='center', fontsize=8)

                st.pyplot(fig)



            st.header("9. Which ChatGPT versions are associated with higher/lower sentiment?")

            # Clean and filter versions
            df['version'] = df['version'].astype(str).str.strip()
            valid_versions = ['3.0', '3.5', '4.0', '4.1']
            df = df[df['version'].isin(valid_versions)]

            # Compute sentiment counts per version
            version_sentiment = df.groupby(['version', 'sentiment']).size().unstack(fill_value=0)

            # Ensure all versions are present in correct order
            version_sentiment = version_sentiment.reindex(valid_versions)

            # Define sentiment order
            sentiments = ['negative', 'neutral', 'positive']
            colors = {'negative': 'salmon', 'neutral': 'skyblue', 'positive': 'lightgreen'}

            # Create 3 small horizontal bar charts side by side
            cols = st.columns(3)

            for i, sentiment in enumerate(sentiments):
                with cols[i]:
                    st.subheader(f"{sentiment.capitalize()}")

                    sentiment_sorted = version_sentiment[sentiment].sort_values(ascending=False)

                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.barh(sentiment_sorted.index, sentiment_sorted.values, color=colors[sentiment])
                    ax.set_title(f"{sentiment.capitalize()} Reviews", fontsize=11)
                    ax.invert_yaxis()
                    ax.set_xlabel("Count")
                    ax.tick_params(labelsize=9)

                    for j, (value, name) in enumerate(zip(sentiment_sorted.values, sentiment_sorted.index)):
                        ax.text(value - 1, j, str(value), va='center', ha='right', fontsize=9, color='black')

                    st.pyplot(fig)


            st.header("10. What are the most common negative feedback themes?")
            neg_text = ' '.join(df[df['sentiment'] == 'negative']['review'])
            neg_wc = WordCloud(width=800, height=400, background_color='white').generate(neg_text)
            st.image(neg_wc.to_array(), use_container_width=True)