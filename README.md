# SMS_Spam_Dectection

## Overview

This project is a machine learning-based solution to classify SMS messages as **spam** or **ham** (legitimate). The model uses natural language processing (NLP) and machine learning techniques, specifically the **Naive Bayes** classifier, to detect spam messages from a dataset of labeled SMS messages.

## Features

- **Text Preprocessing**: Cleans the input text by converting it to lowercase, removing punctuation, and filtering out basic stopwords.
- **SMS Classification**: Classifies the message into either "spam" or "ham."
- **Model Training**: Uses **Multinomial Naive Bayes** classifier for training the model.
- **Evaluation**: Measures the model's performance with accuracy and a detailed classification report (precision, recall, and F1-score).
- **User Input**: Allows the user to input an SMS message to classify as either spam or ham.

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - `pandas`
  - `scikit-learn`
  - `string`

## Requirements

- Python 3.x
- Required Libraries:
