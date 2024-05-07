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

app = Flask(__name__)

# Load pre-computed data (outside a function for efficiency)
feature_list = np.array(pickle.load(open('featurevector.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

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
