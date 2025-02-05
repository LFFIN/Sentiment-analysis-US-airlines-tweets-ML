# Sentiment-analysis-US-airlines-tweets-ML
# Airline Sentiment Analysis Project

## Overview
This project aims to analyze airline customer sentiment using Twitter data. The goal was to classify tweets as either **positive** or **negative**, helping airlines understand passenger experiences and improve services. The dataset contained thousands of tweets related to major U.S. airlines, labeled with sentiment categories.

## Dataset
- **Source**: Twitter airline sentiment dataset
- **Key Columns Used**:
  - `airline_sentiment`: The sentiment label (positive or negative)
  - `text`: The raw tweet content
- **Preprocessing Steps**:
  - Removed neutral tweets to focus on positive and negative classifications.
  - Cleaned text by removing links, punctuation, stopwords, and special characters.
  - Applied **lemmatization** to simplify words to their root forms.

## Feature Engineering
Since machine learning models cannot process raw text, tweets were converted into numerical form using **TF-IDF vectorization**. Three variations were tested:
1. Keeping words that appeared in **at least five tweets**.
2. Limiting vocabulary to **the 2500 most important words**.
3. Further reducing the vocabulary to **500 words**.

## Machine Learning Models Used
Four different classification models were trained and evaluated:
- **Logistic Regression**: A simple and efficient model often used for text classification.
- **Support Vector Machine (SVM)**: Known for its strong performance in high-dimensional text data.
- **Random Forest**: An ensemble method using multiple decision trees for prediction.
- **Neural Network (MLP)**: A deep learning approach attempting to capture complex patterns in the data.

## Model Evaluation
Each model was evaluated based on three key metrics:
- **Accuracy (M1)**: Measures overall correctness of predictions.
- **F1-Score (M2)**: Balances precision and recall, making it useful for handling imbalanced data.
- **Training Time (M3)**: How long the model takes to learn from the data.

### Results
- **SVM achieved the highest accuracy of 91.57%** and the best F1-score of **77.33%**.
- **Logistic Regression performed well with 90.40% accuracy**, but slightly lower recall than SVM.
- **Random Forest had 89.92% accuracy** but was significantly slower in training.
- **Neural Networks had the lowest accuracy (88.05%) and took the longest time to train (160.68 seconds).**

### Best Model: **Support Vector Machine (SVM)**
SVM was the best-performing model due to its:
- **High Accuracy (91.57%)** – Most correct classifications.
- **Fast Training Time (0.061 seconds)** – Efficient and scalable.
- **Ability to Handle Text Data** – SVM finds the optimal decision boundary in text-based high-dimensional spaces.

## Key Takeaways
- **Data Cleaning is Critical**: Removing noise significantly improved model accuracy.
- **Feature Selection Matters**: TF-IDF helped extract the most relevant words for classification.
- **Complexity Doesn’t Always Win**: Neural Networks were slower and didn’t outperform simpler models like SVM.
- **SVM is Ideal for Text Classification**: It provided the best tradeoff between accuracy and efficiency.

## How to Run This Project
### Prerequisites
- Python 3.x
- Libraries: `pandas`, `numpy`, `nltk`, `sklearn`, `matplotlib`

### Steps
1. **Install required libraries**:
   ```sh
   pip install pandas numpy nltk scikit-learn matplotlib
   ```
2. **Run the script** to train and evaluate models.
   ```sh
   python sentiment_analysis.py
   ```
3. **View results**: The script will print model accuracy and generate bar charts comparing performance.

## Future Improvements
- **Hyperparameter Tuning**: Further optimize SVM’s `C` parameter for better results.
- **Use N-Grams**: Improve feature representation by considering bigrams and trigrams.
- **Test Deep Learning Models**: Experiment with LSTMs or transformers for better long-term text understanding.

---
**Author**: Larbi F
**Date**: Jan 2025 
**Purpose**: Machine learning project for sentiment analysis using Twitter airline data.

