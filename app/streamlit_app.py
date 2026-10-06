from pathlib import Path
import sys

import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.recommender import hidden_gems, load_model, recommend

st.set_page_config(page_title="Audible Insights", page_icon="📚", layout="wide")
st.title("📚 Audible Insights")
st.caption("Find your next audiobook using content similarity, quality signals, and catalog insights.")

bundle = load_model()
books = bundle["books"]

with st.sidebar:
    st.header("Explore")
    mode = st.radio("Choose a view", ["Recommendations", "Hidden gems", "Catalog overview"])

if mode == "Recommendations":
    query = st.selectbox("Choose a book you enjoyed", books["title"].sort_values().tolist())
    n = st.slider("Number of recommendations", 3, 10, 5)
    if st.button("Recommend books", type="primary"):
        results = recommend(query, n)
        st.subheader(f"Because you liked {query}")
        st.dataframe(results, use_container_width=True, hide_index=True)

elif mode == "Hidden gems":
    st.subheader("Highly rated, less obvious picks")
    st.write("These books combine strong ratings with lower review volume, so they are less dominated by blockbuster popularity.")
    st.dataframe(hidden_gems(15), use_container_width=True, hide_index=True)

else:
    st.subheader("Catalog overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Books", f"{len(books):,}")
    c2.metric("Authors", f"{books['author'].nunique():,}")
    c3.metric("Average rating", f"{books['rating'].mean():.2f}")
    fig = px.histogram(books, x="rating", nbins=20, title="Rating distribution")
    st.plotly_chart(fig, use_container_width=True)
    top_authors = books.groupby("author", as_index=False)["review_count"].sum().nlargest(15, "review_count")
    fig2 = px.bar(top_authors, x="review_count", y="author", orientation="h", title="Authors by total review volume")
    st.plotly_chart(fig2, use_container_width=True)
