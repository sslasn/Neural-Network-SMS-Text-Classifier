import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

df_train = pd.read_csv("sms_train.csv")
df_test = pd.read_csv("sms_test.csv")
df = pd.concat([df_train, df_test])
df = pd.get_dummies(df, columns=['categorical_column1', 'categorical_column2'])
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['message'])
X_train = X[:len(df_train)]
X_test = X[len(df_train):]
y_train = df_train['label']
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
probabilities = classifier.predict_proba(X_test)
labels = classifier.predict(X_test)

def predict_message(message):
    message_vector = vectorizer.transform([message])
    spam_probability = probabilities[0][1]
    label = "spam" if spam_probability > 0.5 else "ham"
    return [spam_probability, label]
test_message = "Get a free iPhone now!"
prediction = predict_message(test_message)
print("Probability:", prediction[0])
print("Label:", prediction[1])
