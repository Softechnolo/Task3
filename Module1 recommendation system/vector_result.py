# vector_result.py
from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import pickle
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import tensorflow
import cv2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from PIL import Image
import pytesseract

app = Flask(__name__)

# Load pre-computed data (outside a function for efficiency)
feature_list = np.array(pickle.load(open('featurevector.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

# Load dataset
df = pd.read_csv('cleaned_data.csv')

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Fill NaN values with an empty string
df['Description'] = df['Description'].fillna('')

# Vectorize product descriptions
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Description'])

def extract_features(img_path):
  img = cv2.imread(img_path)
  img = cv2.resize(img, (224, 224))
  img = np.array(img)
  expanded_img = np.expand_dims(img, axis=0)
  preprocessed_img = preprocess_input(expanded_img)
  features = model.predict(preprocessed_img).flatten()
  normalized_features = features / norm(features)
  return normalized_features

def recommend(features):
  neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
  neighbors.fit(feature_list)
  distances, indices = neighbors.kneighbors([features])
  return indices.flatten()[1:]  # Exclude the first element (query image)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']

        # Process query using TF-IDF
        tfidf_query = vectorizer.transform([query])

        # Compute cosine similarity between query and all products
        cosine_sim = cosine_similarity(tfidf_query, tfidf_matrix).flatten()

        # Get top 5 most similar products
        top_indices = cosine_sim.argsort()[:-6:-1]
        recommendations = df.iloc[top_indices]

        # Generate description paragraph
        top_sentences = heapq.nlargest(5, zip(cosine_sim, df['Description']))
        description_paragraph = ' '.join(sentence for sim, sentence in top_sentences)

        return render_template('vector_result.html', query=query, description_paragraph=description_paragraph, recommendations=recommendations.to_dict(orient='records'))
    
    return render_template('index.html')  # Create an index.html form where users can enter their search query

@app.route('/ocr', methods=['GET', 'POST'])
def ocr():
    if request.method == 'POST':
        # Check if an image file was uploaded
        image = request.files.get('image')
        if image:
            # Use pytesseract to extract text from the image
            img = Image.open(image)
            query = pytesseract.image_to_string(img)

            # Process query using TF-IDF
            tfidf_query = vectorizer.transform([query])

            # Compute cosine similarity between query and all products
            cosine_sim = cosine_similarity(tfidf_query, tfidf_matrix).flatten()

            # Get top 5 most similar products
            top_indices = cosine_sim.argsort()[:-6:-1]
            recommendations = df.iloc[top_indices]

            # Generate description paragraph
            top_sentences = heapq.nlargest(5, zip(cosine_sim, df['Description']))
            description_paragraph = ' '.join(sentence for sim, sentence in top_sentences)

            return render_template('vector_result.html', query=query, description_paragraph=description_paragraph, recommendations=recommendations.to_dict(orient='records'))
    
    return render_template('index.html')  # Create an index.html form where users can enter their search query

@app.route('/cnn', methods=['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
    uploaded_file = request.files['image']
    if uploaded_file.filename != '':
      try:
        # Save the file
        filepath = os.path.join('static/uploads', uploaded_file.filename)
        uploaded_file.save(filepath)

        # Extract features
        features = extract_features(filepath)

        # Recommend similar images
        recommended_indices = recommend(features)

        # Prepare data for template
        recommended_images = [filenames[i] for i in recommended_indices]
        return render_template('cnn.html', uploaded_image=uploaded_file.filename, recommended_images=recommended_images)
      except Exception as e:
        return 'Error uploading file: ' + str(e)
    else:
      return 'No file selected!'
  return render_template('cnn.html')

if __name__ == '__main__':
  app.run(debug=True)
