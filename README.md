
# Financial News Sentiment Analysis with FinBERT

This project uses **FinBERT** (a BERT model fine-tuned for financial sentiment) to classify Vietnamese financial news articles into **Positive**, **Neutral**, or **Negative** sentiments. It combines **NLP techniques** with **stock market data** to analyze the relationship between news sentiment and stock/index movements.

## Overview

- Crawl and clean news data from [CafeF](https://cafef.vn) and [Vietstock](https://vietstock.vn)
- Fine-tune FinBERT (via HuggingFace) on Vietnamese financial news labeled by stock/index changes
- Compare performance of FinBERT with SVM and MLPClassifier
- Visualize sentiment and market correlation using a Streamlit web app

---

## Tech Stack

- **Language**: Python
- **NLP Libraries**: `transformers`, `NLTK`, `scikit-learn`
- **Visualization**: `Streamlit`, `matplotlib`, `plotly`
- **Others**: `pandas`, `vnstock`, `joblib`, `Git`

---

## Project Structure

```
finbert-news-sentiment/
├── data/                   # Raw and processed financial news data
├── notebooks/              # Jupyter notebooks for EDA & modeling
├── app/                    # Streamlit app
│   ├── main.py
├── models/                 # Saved models and label encoders
├── utils/                  # Custom preprocessing and labeling functions
├── requirements.txt
└── README.md
```

---

## Example Output

| Headline (translated)                               | Predicted Sentiment |
|-----------------------------------------------------|----------------------|
| "Vietcombank announces record quarterly profit"     | Positive             |
| "VNIndex drops after international rate hike"       | Negative             |
| "Construction firms face raw material shortages"    | Neutral              |

---

## Installation

```bash
git clone https://github.com/t-i-n2003/finbert-news-sentiment.git
cd finbert-news-sentiment
pip install -r requirements.txt
```

---

## Run Streamlit App

```bash
cd app
streamlit run main.py
```

---

## Future Improvements

- Integrate RAG and Vector DB for context-aware sentiment
- Improve Vietnamese-specific tokenization (e.g., with VnCoreNLP)
- Deploy via Docker or HuggingFace Spaces

---

## License

This project is open-sourced under the MIT License.

---

## Author

**Tín Nguyễn**  
- [ntinit2003@gmail.com](mailto:ntinit2003@gmail.com)  
- [LinkedIn](https://www.linkedin.com/in/tin-nguyen-04a86a278)  
- [GitHub](https://github.com/t-i-n2003)
