import joblib
from processing.prepare_data import translate_vi_to_en

def predict(text):
    # Load the model
    model = joblib.load('models/svc.pkl')
    # Preprocess the input text
    en_text = translate_vi_to_en(text)
    vectorizer = joblib.load(f'models/tfidf_vectorizer.pkl')
    en_text = vectorizer.transform([en_text]).toarray()
    # Predict the sentiment
    prediction = model.predict(en_text)
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map[prediction[0]]