import os
import pandas as pd
import re
import string
import nltk
import pickle
import urllib.request
import io
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def main():
    # Setup directories
    os.makedirs('dataset', exist_ok=True)
    os.makedirs('model', exist_ok=True)

    csv_path = os.path.join('dataset', 'spam.csv')
    if not os.path.exists(csv_path):
        print("Dataset not found locally. Initiating download...")
        url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                data = response.read()
            df = pd.read_csv(io.BytesIO(data), sep='\t', header=None, names=['Label', 'Message'])
            # Clean labels: map 'ham' -> 'Ham', 'spam' -> 'Spam'
            df['Label'] = df['Label'].map({'ham': 'Ham', 'spam': 'Spam'})
            # Save as CSV
            df.to_csv(csv_path, index=False)
            print(f"Dataset successfully downloaded and saved to {csv_path}")
        except Exception as e:
            print(f"Error downloading from primary URL: {e}")
            try:
                print("Trying fallback URL...")
                alternative_url = "https://raw.githubusercontent.com/lds-trio/sms-spam-collection/master/SMSSpamCollection"
                req = urllib.request.Request(alternative_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=15) as response:
                    data = response.read()
                df = pd.read_csv(io.BytesIO(data), sep='\t', header=None, names=['Label', 'Message'])
                df['Label'] = df['Label'].map({'ham': 'Ham', 'spam': 'Spam'})
                df.to_csv(csv_path, index=False)
                print(f"Dataset successfully downloaded from fallback and saved to {csv_path}")
            except Exception as ex:
                print(f"Fallback download also failed: {ex}")
                raise ex
    else:
        print(f"Dataset already exists at {csv_path}")

    # Download NLTK resources (should be fast if already present)
    print("Checking NLTK resources...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')

    # Load dataset
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded. Total rows: {df.shape[0]}")
    print(df['Label'].value_counts())

    # Initialize Stemmer and Stopwords
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        if not isinstance(text, str):
            return ""
        # 1. Convert to Lowercase
        text = text.lower()
        # 2. Remove Punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # 3. Tokenization
        words = word_tokenize(text)
        # 4. Remove Stopwords & 5. Stemming
        cleaned_words = [ps.stem(word) for word in words if word not in stop_words]
        return " ".join(cleaned_words)

    # Apply preprocessing
    print("Preprocessing text (lowercase, punctuation, tokenization, stopwords, stemming)...")
    df['Clean_Message'] = df['Message'].apply(preprocess_text)

    # Feature Engineering (TF-IDF)
    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Clean_Message'])
    y = df['Label']

    # Train-Test Split (80% Train, 20% Test)
    print("Splitting dataset (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train Model (Multinomial Naive Bayes)
    print("Training Multinomial Naive Bayes classifier...")
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Model Evaluation
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='Spam')
    recall = recall_score(y_test, y_pred, pos_label='Spam')
    f1 = f1_score(y_test, y_pred, pos_label='Spam')

    print("\n" + "="*40)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision (Spam): {precision * 100:.2f}%")
    print(f"Recall (Spam): {recall * 100:.2f}%")
    print(f"F1 Score (Spam): {f1 * 100:.2f}%")
    print("="*40)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Model Saving
    print("\nSaving model and vectorizer...")
    with open(os.path.join('model', 'spam_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join('model', 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    print("Successfully saved 'model/spam_model.pkl' and 'model/vectorizer.pkl'.")

if __name__ == '__main__':
    main()
