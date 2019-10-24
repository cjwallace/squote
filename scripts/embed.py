"""
Script to read quote csv and write csv of embeddings.
"""

import pandas as pd
from bert_serving.client import BertClient

# Bert Client must be running locally
bc = BertClient()

# Read quote data
quotes = pd.read_csv('data/quotes_all.csv', sep=';', skiprows=1)

# Compute embeddings
embeddings = bc.encode(quotes.QUOTE.to_list())
quotes['EMBEDDINGS'] = embeddings.tolist()

# Persist to pickle
quotes.to_pickle('data/embedded_quotes.pkl')