import os.path
import pandas as pd
import re
import nltk
import spacy
import numpy as np
from keras.src.layers import TextVectorization
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.utils import resample
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import time

def begin_cleaner(filepath: str, data_range: int=500) -> pd.DataFrame:
    raw_data = pd.read_csv(filepath)
    raw_data = raw_data.drop(columns=['Id',
                                      'ProductId',
                                      'UserId',
                                      'ProfileName',
                                      'HelpfulnessNumerator',
                                      'HelpfulnessDenominator',
                                      'Time'])
    raw_data = raw_data.head(data_range - 1)
    # raw_data = raw_data.sample(data_range - 1)

    raw_data["holistic_reviews"] = raw_data["Summary"].astype(str) + " " + raw_data["Text"].astype(str)

    raw_data["holistic_reviews"] = raw_data["holistic_reviews"].apply(sanitize_data)
    raw_data["Score"] = pd.to_numeric(raw_data["Score"], errors='coerce')

    clean_data = raw_data.dropna(subset=["Score"])

    X = clean_data["holistic_reviews"].values
    y = clean_data["Score"].values

    y = (y >= 3).astype(int)

    return X, y, clean_data

def sanitize_data(uncleaned_data: str):
    if not isinstance(uncleaned_data, str):
        uncleaned_data = ""

    clean_data = uncleaned_data.lower()
    clean_data = re.sub(r"\n", " ", clean_data)
    clean_data = re.sub(r"\s+", " ", clean_data)
    clean_data = clean_data.strip()

    tokens = word_tokenize(clean_data)
    filtered_tokens_alpha = [word for word in tokens if word.isalpha()]

    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in filtered_tokens_alpha if word not in stop_words]

    # print(filtered_tokens)

    return " ".join(filtered_tokens)

def build_lstm_model(vectorize_layer, embedding_dim, max_length):
    # api stuff
    model = Sequential([
        vectorize_layer,
        Embedding(input_dim=5000 + 1, output_dim=embedding_dim, mask_zero=True),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        Dropout(0.5),
        Dense(24, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model

def downsample_majority(X, y):
    df = pd.DataFrame({'text': X, 'label': y})

    majority = df[df.label == 1]
    minority = df[df.label == 0]

    if len(majority) < len(minority):
        majority, minority = minority, majority

    print(f"\nBefore downsampling:")
    print(f"Majority class: {len(majority)} samples")
    print(f"Minority class: {len(minority)} samples")

    majority_downsampled = resample(majority,
                                       replace=False,
                                       n_samples=len(minority),
                                       random_state=42)

    balanced = pd.concat([majority_downsampled, minority])

    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nAfter downsampling:")
    print(f"Total samples: {len(balanced)}")
    print(f"Class 0: {len(balanced[balanced.label == 0])}")
    print(f"Class 1: {len(balanced[balanced.label == 1])}")

    return balanced['text'].values, balanced['label'].values

def train_model(X, y):

    print("=== Class Distribution ===")
    '''
    This ranks the balance of the dataset, i.e 48% negative, 52% positive
    '''
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = "Positive" if label == 1 else "Negative"
        print(f"{label_name}: {count} ({count / len(y) * 100:.1f}%)")

    X_balanced, y_balanced = downsample_majority(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )

    max_tokens = 5000
    max_length = 100
    embedding_dim = 256

    # api stuff
    vectorize_layer = TextVectorization(
        max_tokens=max_tokens,
        output_mode='int',
        output_sequence_length=max_length
    )

    vectorize_layer.adapt(X_train)

    selected_model = build_lstm_model(vectorize_layer, embedding_dim, max_length)
    print("\n" + "=" * 50)
    print(selected_model.summary())
    print("=" * 50 + "\n")

    # api stuff
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    start_timer = time.time()

    history = selected_model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1,
    )

    end_timer = time.time()
    print(f"\nTraining completed in {end_timer - start_timer:.2f} seconds")

    y_pred_probs = selected_model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    print("\n=== Prediction Distribution ===")
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    for label, count in zip(unique_pred, counts_pred):
        label_name = "Positive" if label == 1 else "Negative"
        print(f"Predicted {label_name}: {count} ({count / len(y_pred) * 100:.1f}%)")


    '''
    bunch of calls to libraries
    '''
    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return selected_model, vectorize_layer, history

def test_model(review_text, model):
    cleaned = sanitize_data(review_text)

    pred = model.predict(tf.constant([cleaned]))[0][0]

    sentiment = "Positive" if pred > 0.5 else "Negative"
    confidence = pred if pred > 0.5 else 1 - pred

    return sentiment, confidence

def trainer_function(model, reviews):
    print("\n=== Test Predictions ===")
    for review in reviews:
        sentiment, confidence = test_model(review, model)
        print(f"\nReview: {review}")
        print(f"Sentiment: {sentiment} (confidence: {confidence:.2%})")

def export_cleaned_set(data: pd.DataFrame, file_name: str):
    data.to_csv(file_name, index=False)
    abs_path = os.path.abspath(file_name)
    print(f"File successfully exported as {file_name} to {abs_path}.")


if __name__ == "__main__":
    X,y, cleaned_data = begin_cleaner('Reviews.csv')
    model, vectorize_layer, history = train_model(X,y)

    test_reviews = [
        "This is the literal best candy I've ever had. I have ascended into the next 4 mortal planes of existence thanks to this.",
        "I would just like to kiss the maker of this dog food, I love eating it in front of my dog.",
        "this is a negative review",
        "To the CEO of this product. This is so so bad that I will be stringing you up by your eyelids and stoning you to death.",
        "i didn't like it it tasted crap",
        "this is awful",
        "prety fire my g! two two's my word",
        "i don't like this it taste bad"
    ]

    trainer_function(model, test_reviews)

    export_cleaned_set(cleaned_data, 'cleaned_reviews.csv')
