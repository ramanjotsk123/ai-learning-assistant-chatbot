# AI Learning Assistant Chatbot

## Project Overview

An AI-focused learning assistant that retrieves relevant
answers to questions about Artificial Intelligence,
Machine Learning, Python, Data Science, and study skills.

## Objective

Explore Natural Language Processing and semantic search
using a pretrained transformer-based sentence embedding model.

## Technologies

- Python
- Streamlit
- Sentence Transformers
- NumPy
- Pretrained model: all-MiniLM-L6-v2

## How It Works

1. Store a small collection of prepared questions and answers.
2. Encode the questions into sentence embeddings.
3. Encode the user's question with the same model.
4. Compare the embeddings using vector similarity.
5. Retrieve and display the answer associated with
   the closest matching question.

## Features

- Interactive question input
- Semantic question matching
- Prepared answers across AI and programming topics
- Reusable model loading through Streamlit caching

## Run Locally

Install the dependencies:

    pip install -r requirements.txt

Start the app:

    streamlit run app.py

## Limitations

This prototype retrieves existing answers rather than
generating new ones. It may return an irrelevant answer
for questions outside its knowledge base.

The model and application have not been independently
tested as part of this initial GitHub upload.

## Model Reference

https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2.
