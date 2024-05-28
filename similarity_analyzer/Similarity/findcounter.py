import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def detect_rectangles(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    edges = cv2.Canny(img, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangles = []
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            rectangles.append((x, y, w, h))
    
    return rectangles

def normalize_features(rectangles, image_width, image_height):
    normalized_features = []
    for (x, y, w, h) in rectangles:
        norm_x = x / image_width
        norm_y = y / image_height
        norm_w = w / image_width
        norm_h = h / image_height
        normalized_features.append((norm_x, norm_y, norm_w, norm_h))
    return normalized_features

def create_feature_vector(rectangles, image_width, image_height):
    normalized_features = normalize_features(rectangles, image_width, image_height)
    num_rectangles = len(rectangles)
    
    # Flatten normalized features
    flattened_features = np.array(normalized_features).flatten()
    
    # Combine number of rectangles with flattened features
    feature_vector = np.concatenate(([num_rectangles], flattened_features))
    
    return feature_vector

def calculate_similarity(vector1, vector2):
    # Normalize the lengths of the feature vectors by padding with zeros
    max_length = max(len(vector1), len(vector2))
    padded_vector1 = np.pad(vector1, (0, max_length - len(vector1)), 'constant')
    padded_vector2 = np.pad(vector2, (0, max_length - len(vector2)), 'constant')
    
    # Calculate Euclidean distance
    euclidean_dist = euclidean(padded_vector1, padded_vector2)
    
    # Calculate Cosine similarity
    cosine_sim = cosine_similarity([padded_vector1], [padded_vector2])[0][0]
    
    return euclidean_dist, cosine_sim

def plot_vectors(vector1, vector2):
    # Normalize the lengths of the feature vectors by padding with zeros
    max_length = max(len(vector1), len(vector2))
    padded_vector1 = np.pad(vector1, (0, max_length - len(vector1)), 'constant')
    padded_vector2 = np.pad(vector2, (0, max_length - len(vector2)), 'constant')
    
    # Use PCA to reduce dimensions to 2D for visualization
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform([padded_vector1, padded_vector2])
    
    plt.figure(figsize=(8, 6))
    plt.quiver(0, 0, vectors_2d[0, 0], vectors_2d[0, 1], angles='xy', scale_units='xy', scale=1, color='r', label='Vector 1')
    plt.quiver(0, 0, vectors_2d[1, 0], vectors_2d[1, 1], angles='xy', scale_units='xy', scale=1, color='b', label='Vector 2')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axhline(0, color='grey', lw=0.5)
    plt.axvline(0, color='grey', lw=0.5)
    plt.grid()
    plt.legend()
    plt.title('2D PCA of Feature Vectors')
    plt.savefig('/code/Img/plot_vectors.jpg')

def plot_similarity_scores(euclidean_dist, cosine_sim):
    labels = ['Euclidean Distance', 'Cosine Similarity']
    scores = [euclidean_dist, cosine_sim]
    
    plt.figure(figsize=(8, 6))
    plt.bar(labels, scores, color=['red', 'blue'])
    plt.title('Similarity Scores')
    plt.savefig('/code/Img/plot_similarity_scores.jpg')