import os
import pickle
import string
import io
import urllib.request
import nltk
import pandas as pd
from flask import Flask, request, jsonify, render_template
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# ─── NLTK Setup ──────────────────────────────────────────────────────────────
nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

for pkg in ['stopwords', 'punkt', 'punkt_tab']:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH      = os.path.join('model', 'spam_model.pkl')
VECTORIZER_PATH = os.path.join('model', 'vectorizer.pkl')
DATASET_PATH    = os.path.join('dataset', 'spam.csv')

os.makedirs('model',   exist_ok=True)
os.makedirs('dataset', exist_ok=True)

# ─── Auto-train if model not present ─────────────────────────────────────────
def train_and_save():
    """Download dataset, train Naive Bayes, save model & vectorizer."""
    print("[INFO] Model not found. Starting auto-training...")

    # Download dataset
    url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as response:
        data = response.read()

    df = pd.read_csv(io.BytesIO(data), sep='\t', header=None, names=['Label', 'Message'])
    df['Label'] = df['Label'].map({'ham': 'Ham', 'spam': 'Spam'})
    df.to_csv(DATASET_PATH, index=False)
    print(f"[INFO] Dataset downloaded: {df.shape[0]} rows")

    ps        = PorterStemmer()
    sw        = set(stopwords.words('english'))

    def preprocess(text):
        if not isinstance(text, str):
            return ""
        text  = text.lower().translate(str.maketrans('', '', string.punctuation))
        words = word_tokenize(text)
        return " ".join([ps.stem(w) for w in words if w not in sw])

    df['clean'] = df['Message'].apply(preprocess)

    vec   = TfidfVectorizer()
    X     = vec.fit_transform(df['clean'])
    y     = df['Label']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    mdl = MultinomialNB()
    mdl.fit(X_train, y_train)

    with open(MODEL_PATH,      'wb') as f: pickle.dump(mdl, f)
    with open(VECTORIZER_PATH, 'wb') as f: pickle.dump(vec, f)
    print("[INFO] Model and vectorizer saved successfully.")
    return mdl, vec


if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    model, vectorizer = train_and_save()
else:
    with open(MODEL_PATH,      'rb') as f: model      = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f: vectorizer = pickle.load(f)
    print("[INFO] Model loaded from disk.")

# ─── Preprocessing (same as training) ────────────────────────────────────────
ps         = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text  = text.lower().translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    return " ".join([ps.stem(w) for w in words if w not in stop_words])

# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(silent=True)
        if not data or 'message' not in data:
            return jsonify({'error': 'No message text provided.'}), 400

        cleaned   = preprocess_text(data['message'])
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probs      = dict(zip(model.classes_, model.predict_proba(vectorized)[0]))
        confidence = probs[prediction]

        return jsonify({
            'prediction':   prediction,
            'confidence':   float(confidence),
            'probabilities': {k: float(v) for k, v in probs.items()}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
