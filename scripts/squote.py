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

    # change dtype in place for memory efficiency
    quotes['EMBEDDINGS'] = quotes['EMBEDDINGS'].apply(
        lambda arr: np.array(arr, dtype='float32')
    )

    quote_embeddings = np.stack(quotes.EMBEDDINGS.values)

    # reduce memory footprint by dropping column
    quotes.drop('EMBEDDINGS', axis='columns')

    # normalize embeddings for cosine distance
    embedding_sums = quote_embeddings.sum(axis=1)
    normed_embeddings = quote_embeddings / embedding_sums[:, np.newaxis]
    return quotes, normed_embeddings


@st.cache(allow_output_mutation=True)
def create_index(embeddings):
    """
    Create an index over the quote embeddings for fast similarity search.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


quotes, embeddings = load_quotes_and_embeddings()

index = create_index(embeddings)

bc = BertClient()


text = st.text_area(
    "Your text",
    "I dreamed a dream."
)

if not text:
    text = "Emptiness"

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

    Made by [chrjs](https://chrjs.io).
    """
)

bc.close()
