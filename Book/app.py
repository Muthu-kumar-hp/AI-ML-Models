# book_recommender.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# ---------- Setup ----------
st.set_page_config(page_title="Book Recommender", layout="centered")

# ---------- Load Data ----------
def load_data():
    books = pd.DataFrame({
        'Book_ID': range(1, 11),
        'Title': ['The Great Gatsby', 'To Kill a Mockingbird', '1984', 'Pride and Prejudice',
                  'The Catcher in the Rye', 'Lord of the Flies', 'The Hobbit', 'Fahrenheit 451',
                  'Jane Eyre', 'Wuthering Heights'],
        'Author': ['F. Scott Fitzgerald', 'Harper Lee', 'George Orwell', 'Jane Austen',
                   'J.D. Salinger', 'William Golding', 'J.R.R. Tolkien', 'Ray Bradbury',
                   'Charlotte Brontë', 'Emily Brontë'],
        'Genre': ['Classic', 'Drama', 'Sci-Fi', 'Romance', 'Coming of Age', 'Adventure',
                  'Fantasy', 'Sci-Fi', 'Romance', 'Romance']
    })

    ratings = pd.DataFrame({
        'UserID': np.random.randint(1, 51, 300),
        'BookID': np.random.randint(1, 11, 300),
        'Rating': np.random.randint(1, 6, 300)
    })

    return books, ratings

books_df, ratings_df = load_data()

# ---------- Recommender Functions ----------
def content_based(book_title, n=3):
    books_df['features'] = books_df['Title'] + " " + books_df['Author'] + " " + books_df['Genre']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(books_df['features'])
    cosine_sim = cosine_similarity(tfidf_matrix)

    idx = books_df[books_df['Title'] == book_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]

    return books_df.iloc[[i[0] for i in sim_scores]][['Title', 'Author', 'Genre']]

def collaborative(book_title, n=3):
    merged = ratings_df.merge(books_df, left_on='BookID', right_on='Book_ID')
    user_item = merged.pivot_table(index='UserID', columns='Title', values='Rating').fillna(0)

    if book_title not in user_item.columns:
        return pd.DataFrame()

    model = NearestNeighbors(metric='cosine')
    model.fit(user_item.T)
    book_vec = user_item[book_title].values.reshape(1, -1)
    distances, indices = model.kneighbors(book_vec, n_neighbors=n+1)

    result = []
    for i in range(1, len(indices[0])):
        title = user_item.columns[indices[0][i]]
        book = books_df[books_df['Title'] == title].iloc[0]
        result.append({'Title': title, 'Author': book['Author'], 'Genre': book['Genre']})

    return pd.DataFrame(result)

def hybrid(book_title, n=3):
    content = content_based(book_title, n * 2)
    collab = collaborative(book_title, n * 2)

    scores = {}
    for t in content['Title']:
        scores[t] = scores.get(t, 0) + 0.6
    for t in collab['Title']:
        scores[t] = scores.get(t, 0) + 0.4

    top_titles = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]

    result = []
    for title, score in top_titles:
        book = books_df[books_df['Title'] == title].iloc[0]
        result.append({'Title': title, 'Author': book['Author'], 'Genre': book['Genre'], 'Score': round(score * 100, 1)})

    return pd.DataFrame(result)

# ---------- UI ----------
st.markdown("<h1 style='text-align: center;'>📖 Book Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("---")

book_title = st.selectbox("📚 Select a Book:", [''] + list(books_df['Title']))
rec_type = st.radio("🔍 Choose Recommendation Type:", ["📘 Content-Based", "👥 Collaborative", "🧪 Hybrid"])

recommend = st.button("🔎 Get Recommendations")

# ---------- Recommendation Output ----------
if recommend:
    if not book_title:
        st.warning("⚠️ Please select a book first!")
    else:
        if rec_type == "📘 Content-Based":
            st.subheader(f"📘 Content-Based Recommendations for *'{book_title}'*")
            recs = content_based(book_title)
        elif rec_type == "👥 Collaborative":
            st.subheader(f"👥 Collaborative Recommendations for *'{book_title}'*")
            recs = collaborative(book_title)
        else:
            st.subheader(f"🧪 Hybrid Recommendations for *'{book_title}'*")
            recs = hybrid(book_title)

        if recs.empty:
            st.info("No recommendations available.")
        else:
            for _, row in recs.iterrows():
                st.markdown(f"**📖 {row['Title']}**  \n✍️ Author: *{row['Author']}*  \n🏷️ Genre: _{row['Genre']}_")
                if 'Score' in row:
                    st.progress(int(row['Score']))
                st.markdown("---")

# ---------- Footer ----------
st.markdown("© 2025 BookRecommender | Built with ❤️ using Streamlit", unsafe_allow_html=True)
