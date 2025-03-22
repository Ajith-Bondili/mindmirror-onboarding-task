# MindMirror Onboarding Assignment

## Introduction

This project implements two core AI-powered features for a journaling app:

- **Emotion Detection:** Automatically identifies the underlying emotions in journal entries.
- **Text Summarization:** Generates concise summaries that capture both the key points and emotional tone of user entries.

This assignment provided hands-on experience with Natural Language Processing (NLP) techniques using pre-trained models and the Hugging Face pipeline API.

## Setup and Installation

1. **Clone the Repository**  
   Ensure you have the project folder cloned locally.

2. **Install Dependencies**  
   Create a virtual environment (optional but recommended) and install the Python libraries:
   ```bash
   pip install -r submissions/ajith/requirements.txt
   ```

3. **Download NLTK Data**  
   The preprocessing code automatically downloads necessary NLTK data when run.

## Task 1: Emotion Detection

- **Model Choice:**  
  The project uses `j-hartmann/emotion-english-distilroberta-base` for emotion classification. This model is efficient, fast, and trained on the GoEmotions dataset supporting 28 emotion classes.

- **Implementation Overview:**  
  - Journal entries are read from `journals.json` and preprocessed using the functions from `preprocessing.py`.
  - Each journal entry is segmented and passed through an emotion detection pipeline.
  - Detected emotions (filtered by a confidence threshold) are saved to `emotion_detection_results.json`.

## Task 2: Text Summarization

- **Model Choice:**  
  The summarization task utilizes the `facebook/bart-large-cnn` model. This model is well-suited for summarizing long texts, capturing key ideas, and offers control over summary length.

- **Implementation Overview:**  
  - Each journal entry is preprocessed and split into manageable segments if needed.
  - The summarization pipeline processes each segment, and the results are combined with concise connecting phrases.
  - Summaries are saved to `journal_summaries.txt`.

## Running the Code

- **Emotion Detection:**  
  Run the emotion detection script with:
  ```bash
  python submissions/ajith/task1.py
  ```
  The results will be written to `emotion_detection_results.json`.

- **Text Summarization:**  
  Run the summarization script with:
  ```bash
  python submissions/ajith/task2.py
  ```
  The output summaries will be saved in `journal_summaries.txt`.

## Reflections and Observations

See `reflections.txt` for a detailed account of the challenges encountered and observations made during the assignment. 
