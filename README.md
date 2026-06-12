# Spam Email Classifier using Machine Learning

A machine learning-based web application that automatically classifies emails and SMS messages as **Spam** or **Not Spam (Ham)** using Natural Language Processing (NLP) and Naive Bayes Classification.

## Project Structure

```
Spam_Email_Classifier/
│
├── dataset/
│   └── spam.csv          # SMS Spam Collection Dataset
│
├── model/
│   ├── spam_model.pkl    # Serialized Multinomial Naive Bayes model
│   └── vectorizer.pkl    # Serialized TF-IDF Vectorizer
│
├── templates/
│   └── index.html        # Front-end UI (Bootstrap 5 + Custom Glassmorphism UI)
│
├── static/
│   └── style.css         # Styling, gradients, and micro-animations
│
├── app.py                # Flask Backend server
├── train_model.py        # Model training and evaluation script
├── requirements.txt      # Project dependencies
└── README.md             # Documentation
```

## Features

- **Text Preprocessing**: Automated cleaning pipeline including lowercasing, punctuation removal, tokenization, stopword filtering, and Porter stemming.
- **Machine Learning Classifier**: Multinomial Naive Bayes trained on the SMS Spam Collection dataset, achieving **96.59% accuracy**.
- **Interactive UI**: Clean, modern, responsive glassmorphic dashboard built using Bootstrap 5 with real-time AJAX requests, dynamic badges, progress bars, confidence score visualizations, and clickable template test shortcuts.

## Prerequisites

- **Python 3.11+**
- **pip** (Python package installer)

## Installation & Setup

1. **Clone or Navigate to the Workspace Directory**:
   ```bash
   cd Spam_Email_Classifier
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model**:
   Run the training script to download the SMS Spam Collection corpus, run preprocessing, train the Naive Bayes model, evaluate its accuracy, and save the serialized weights under `model/`.
   ```bash
   python train_model.py
   ```

4. **Launch the Web Server**:
   Start the Flask backend application:
   ```bash
   python app.py
   ```

5. **Open in Browser**:
   Open your browser and navigate to `http://127.0.0.1:5000` to interact with the classifier.
