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

"""
Enter your text and get some questionably relevant,
questionably famous quotes.
"""

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

# Create an index for fast vector search
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

text = st.text_area(
    "Your text",
    "I have a dream."
)

# normalize embedding for cosine distance
text_embedding = bc.encode([text])
normalized_text_embedding = text_embedding / text_embedding.sum()

_, idx = index.search(normalized_text_embedding, 5)

relevant_quotes = quotes.iloc[idx.flatten()].QUOTE.values
relevant_authors = quotes.iloc[idx.flatten()].AUTHOR.values

st.subheader("Your quotes")

for q in range(5):
    st.markdown('>'+relevant_quotes[q])
    st.text(relevant_authors[q])

st.subheader("What is this?")

st.markdown(
    """
    It's semantic search powered by:
    - [bert-as-a-service](https://github.com/hanxiao/bert-as-service)
        &larr; the math
    - [faiss](https://github.com/facebookresearch/faiss)
        &larr; the search
    - [streamlit](https://github.com/streamlit/streamlit)
        &larr; the interface

    The repo is hosted on GitHub at [cjwallace/squote](https://github.com/cjwallace/squote).

    Made by [chrjs](chrjs.io).
    """
)
