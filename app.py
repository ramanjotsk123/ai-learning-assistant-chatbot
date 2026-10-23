import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# Small knowledge base for the learning assistant.
# The bot retrieves answers from these prepared entries.
KNOWLEDGE_BASE = [
    {
        "question": "What is artificial intelligence?",
        "topic": "AI Fundamentals",
        "answer": (
            "Artificial Intelligence (AI) is a field of computing "
            "that develops systems capable of tasks such as language "
            "understanding, pattern recognition, reasoning, and prediction."
        ),
    },
    {
        "question": "What is machine learning?",
        "topic": "Machine Learning",
        "answer": (
            "Machine Learning is a branch of AI in which algorithms "
            "learn patterns from data and use those patterns to make "
            "predictions or decisions. Examples include regression, "
            "classification, and clustering."
        ),
    },
    {
        "question": "What is data science?",
        "topic": "Data Science",
        "answer": (
            "Data Science combines programming, statistics, and "
            "domain knowledge to extract useful insights from data. "
            "Common tools include Python, Pandas, NumPy, and visualisation libraries."
        ),
    },
    {
        "question": "What is natural language processing?",
        "topic": "Natural Language Processing",
        "answer": (
            "Natural Language Processing (NLP) is an area of AI "
            "that helps computers process and analyse human language. "
            "Examples include text classification, search, translation, "
            "and chatbots."
        ),
    },
    {
        "question": "What is a transformer model?",
        "topic": "Deep Learning",
        "answer": (
            "A transformer is a neural-network architecture that uses "
            "attention mechanisms to process relationships between "
            "tokens. Transformer-based models are widely used in modern NLP."
        ),
    },
    {
        "question": "What are sentence embeddings?",
        "topic": "Semantic Search",
        "answer": (
            "Sentence embeddings represent text as numerical vectors. "
            "Texts with similar meanings can have similar vectors, "
            "allowing a system to retrieve relevant information even "
            "when the wording differs."
        ),
    },
    {
        "question": "How does semantic search work?",
        "topic": "Semantic Search",
        "answer": (
            "Semantic search converts a query and stored text into "
            "embeddings, then compares their similarity. It retrieves "
            "text based on meaning rather than relying only on exact "
            "keyword matches."
        ),
    },
    {
        "question": "What is supervised learning?",
        "topic": "Machine Learning",
        "answer": (
            "Supervised learning trains a model using examples that "
            "contain inputs and known target labels or values. "
            "Classification predicts categories, while regression "
            "predicts numerical values."
        ),
    },
    {
        "question": "What is the difference between classification and regression?",
        "topic": "Machine Learning",
        "answer": (
            "Classification predicts a category, such as spam or "
            "not spam. Regression predicts a numerical value, such "
            "as a house price or an estimated examination grade."
        ),
    },
    {
        "question": "What is overfitting in machine learning?",
        "topic": "Model Evaluation",
        "answer": (
            "Overfitting occurs when a model learns the training data "
            "too closely, including noise, and performs poorly on "
            "unseen data. Testing on separate data and controlling "
            "model complexity can help identify or reduce overfitting."
        ),
    },
    {
        "question": "Why do we split data into training and testing sets?",
        "topic": "Model Evaluation",
        "answer": (
            "Training data is used to fit a model. Test data is kept "
            "separate to estimate how the model performs on examples "
            "it did not use during training."
        ),
    },
    {
        "question": "How can I learn Python?",
        "topic": "Python Programming",
        "answer": (
            "Start with variables, data types, conditions, loops, "
            "functions, lists, and dictionaries. Then practise with "
            "small projects and learn libraries such as Pandas and "
            "scikit-learn through practical exercises."
        ),
    },
    {
        "question": "What are Git and GitHub?",
        "topic": "Developer Tools",
        "answer": (
            "Git is a version-control system that records changes "
            "to files. GitHub is an online platform for hosting Git "
            "repositories, documenting projects, and collaborating "
            "with other developers."
        ),
    },
    {
        "question": "How should I debug a Python program?",
        "topic": "Python Programming",
        "answer": (
            "Read the complete error message, identify the line where "
            "the error occurred, check variable values and data types, "
            "and reproduce the problem with a small example. Fix one "
            "issue at a time and test the program again."
        ),
    },
    {
        "question": "How do I write a good GitHub README?",
        "topic": "Portfolio Development",
        "answer": (
            "Explain the project's purpose, features, technologies, "
            "setup instructions, usage, data sources, limitations, "
            "and testing status. Include screenshots or examples "
            "when they help someone understand the project."
        ),
    },
    {
        "question": "What is active recall?",
        "topic": "Study Skills",
        "answer": (
            "Active recall means trying to remember information "
            "without immediately looking at your notes. Practice "
            "questions, flashcards, and explaining a topic from "
            "memory are useful examples."
        ),
    },
    {
        "question": "How does spaced repetition help learning?",
        "topic": "Study Skills",
        "answer": (
            "Spaced repetition revisits material at increasing "
            "intervals. It can help strengthen long-term memory "
            "when combined with active recall and regular practice."
        ),
    },
    {
        "question": "What is a neural network?",
        "topic": "Deep Learning",
        "answer": (
            "A neural network is a machine-learning model made of "
            "connected computational units arranged in layers. "
            "During training, its parameters are adjusted to learn "
            "patterns from examples."
        ),
    },
]


@st.cache_resource
def load_model_and_embeddings():
    """Load the pretrained model and encode the FAQ questions."""
    model = SentenceTransformer(MODEL_NAME)

    questions = [
        item["question"] for item in KNOWLEDGE_BASE
    ]

    question_embeddings = model.encode(
        questions,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return model, question_embeddings


def find_best_answer(question, model, question_embeddings):
    """Retrieve the FAQ question most similar to the user's query."""
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    similarity_scores = question_embeddings @ query_embedding
    best_index = int(np.argmax(similarity_scores))

    return KNOWLEDGE_BASE[best_index]


def main():
    st.set_page_config(
        page_title="AI Learning Assistant",
        page_icon="🤖",
        layout="centered",
    )

    st.title("🤖 AI Learning Assistant")
    st.write(
        "Ask questions about AI, Machine Learning, Python, "
        "Data Science, and study skills."
    )

    st.info(
        "This demo retrieves the closest answer from a prepared "
        "knowledge base. It does not generate new answers, and "
        "its suggestions may not always be relevant."
    )

    question = st.text_input(
        "Enter your question",
        placeholder="For example: How does AI understand text?",
    )

    ask_clicked = st.button(
        "Find an answer",
        type="primary",
    )

    st.caption(
        "Try asking about sentence embeddings, overfitting, "
        "Python, neural networks, or GitHub."
    )

    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question first.")
            return

        try:
            with st.spinner(
                "Loading the language model and searching..."
            ):
                model, question_embeddings = (
                    load_model_and_embeddings()
                )

                result = find_best_answer(
                    question,
                    model,
                    question_embeddings,
                )

            st.subheader("Best matching answer")
            st.write(result["answer"])

            st.caption(f"Topic: {result['topic']}")
            st.caption(
                f"Matched knowledge-base question: "
                f"{result['question']}"
            )

        except Exception as error:
            st.error(
                "The assistant could not load the model or "
                "process this question. Check the installed "
                "dependencies and internet connection."
            )
            st.exception(error)


if __name__ == "__main__":
    main()
