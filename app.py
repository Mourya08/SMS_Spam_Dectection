import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep relevant columns and rename them
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Encode labels: ham = 0, spam = 1
label_encoder = LabelEncoder()
df['label_num'] = label_encoder.fit_transform(df['label'])

# Basic stopwords list
basic_stopwords = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
    "just", "don", "should", "now"
}

# Text preprocessing
def preprocess_text(text):
    text = text.lower()  # lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = ' '.join(word for word in text.split() if word not in basic_stopwords)  # remove stopwords
    return text

df['cleaned_message'] = df['message'].apply(preprocess_text)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_message'], df['label_num'], test_size=0.2, random_state=42)

# Vectorize with TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n", report)
user_input = input("\n Enter an SMS message to classify: ")
cleaned_input = preprocess_text(user_input)
input_vector = vectorizer.transform([cleaned_input])
prediction = model.predict(input_vector)[0]
predicted_label = "Spam" if((label_encoder.inverse_transform([prediction])[0]).upper())=="SPAM" else "LEGITIMATE"

print(f"\n Prediction: This message is **{predicted_label.upper()}**")