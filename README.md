# Airline Tweet Sentiment Analysis with LSTM

This project is a **Sentiment Analysis model** built using **Twitter data about airline services**. It applies **Natural Language Processing (NLP)** techniques and a **Recurrent Neural Network (LSTM)** to classify customer feedback as **positive** or **negative**.

---

##  Features

* **Dataset Preprocessing**:

  * Cleaning tweets (removing URLs, mentions, hashtags, special characters, and stopwords).
  * Converting text to numerical sequences using Keras Tokenizer.
  * Padding sequences for uniform input length.

* **Deep Learning Model**:

  * Embedding layer for word representation.
  * LSTM layer with dropout for sequence learning.
  * Dense layer with sigmoid activation for binary classification.

* **Training**:

  * Trains the model on labeled airline tweets (positive/negative).
  * Uses binary crossentropy loss and Adam optimizer.
  * Tracks accuracy and validation performance across epochs.

* **Evaluation**:

  * Generates a **confusion matrix heatmap** to visualize model performance.
  * Provides accuracy metrics for training and validation.

* **Prediction Function**:

  * Takes custom text input and predicts whether the sentiment is **Positive** or **Negative**.

---

##  Workflow

1. **Data Loading**: Uses `Tweets.csv` containing text and sentiment labels.
2. **Text Preprocessing**: Cleans and prepares raw text for modeling.
3. **Tokenization & Padding**: Converts text into numerical sequences.
4. **Model Training**: LSTM model learns sentiment patterns in tweets.
5. **Model Evaluation**: Visualizes confusion matrix and calculates metrics.
6. **Prediction**: User can input their own sentences for sentiment classification.

---

## Tech Stack

* **Python**
* **Pandas, NumPy** – Data preprocessing
* **NLTK** – Stopwords removal
* **Matplotlib, Seaborn** – Visualization
* **WordCloud** – Text visualization (optional)
* **TensorFlow / Keras** – Deep learning model
* **Scikit-learn** – Confusion matrix

---

## Example Predictions

```python
predict_sentiment("This is the worst flight experience of my life!")  
# Output: Negative  

predict_sentiment("I had a great time, staff were super friendly!")  
# Output: Positive  
```

---

##  Dataset

The project uses the **Airline Tweets dataset** (`Tweets.csv`), which contains tweets labeled as `positive`, `negative`. For this model, only **positive** and **negative** tweets are used.

---

##  Future Improvements

* Add **multi-class classification** (positive, negative, neutral).
* Use **pre-trained embeddings** (GloVe, Word2Vec, BERT).
* Deploy as a **Flask/Django web app** for real-time sentiment analysis.
* Integrate **streaming data from Twitter API** for live analysis.

---

##  License

This project is open-source and available under the MIT License.


