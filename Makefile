.PHONY: dirs data model requirements serve embed app

dirs:
	mkdir data
	mkdir models

data:
	curl https://thewebminer.com/d/quotes_all.csv -o data/quotes_all.csv

model:
	cd models; \
	curl -O https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip; \
	unzip cased_L-12_H-768_A-12.zip

requirements:
	python3 -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt

serve:
	bash scripts/serve.sh

embed:
	python scripts/embed.py

app:
	streamlit run scripts/app.py
