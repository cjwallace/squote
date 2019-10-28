# squote

A semantic search engine that takes some input text and returns some (questionably) relevant (questionably) famous quotes.

![Squote finding relevant quotes](https://user-images.githubusercontent.com/6513950/67687144-bee79b80-f98f-11e9-975f-76c71207f7a0.gif)

Built with:
- [bert-as-a-service](https://github.com/hanxiao/bert-as-service)
- [faiss](https://github.com/facebookresearch/faiss)
- [streamlit](https://github.com/streamlit/streamlit)

Quotes from [https://thewebminer.com/](https://thewebminer.com/).

## setup

First, install the necessary dependencies into a python 3 environment of your choice.
For instance, to install the deps into a venv, run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

All other commands should be run from within the virtual environment.

A Makefile is provided to make things nice and easy.

```bash
make dirs
make data   # downloads the raw quote data
make model  # downloads ~350MB of BERT weights
```

## running

Before we can run the app, we need embeddings of the quotes.
To generate the embeddings and save them in a pickled pandas DataFrame, run the commands below.
This will take some time (couple of hours) on CPU.

```bash
make serve  # this runs bert-as-a-service
make embed  # this computes the embeddings
```

Once the embeddings exist, we can run the streamlit app with:

```bash
make serve  # not needed if still running from above
make app
```

Have fun!
