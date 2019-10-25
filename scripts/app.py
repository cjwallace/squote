"""
Script to find closest quote to given text
"""

import faiss
import pandas as pd
import numpy as np
import streamlit as st

from bert_serving.client import BertClient

st.title("Squote")
st.subheader("Semantic quote search")

"""Enter your text and get a questionably relevant quote :)"""

@st.cache(allow_output_mutation=True)
def load_quotes_and_embeddings():
    quotes = pd.read_pickle('data/embedded_quotes.pkl')
    quote_embeddings = np.stack(quotes.EMBEDDINGS.values).astype('float32')

    # normalize embeddings for cosine distance
    embedding_sums = quote_embeddings.sum(axis=1)
    normed_embeddings = quote_embeddings / embedding_sums[:, np.newaxis]
    return quotes, normed_embeddings

quotes, embeddings = load_quotes_and_embeddings()

bc = BertClient()

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

text = st.text_area(
    "Your text",
    "I have a dream."
)

text_embedding = bc.encode([text])
normalized_text_embedding = text_embedding / text_embedding.sum()

d, i = index.search(normalized_text_embedding, 5)

relevant_quotes = quotes.iloc[i.flatten()].QUOTE.values
relevant_authors = quotes.iloc[i.flatten()].AUTHOR.values

"""Here are five "relevant" quotes:"""

for ix in range(5):
    st.markdown('>'+relevant_quotes[ix])
    st.text(relevant_authors[ix])