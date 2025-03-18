import pandas as pd
import numpy as np

# ==============================
# 1. LOAD & PREPROCESS DATA
# ==============================

# Load datasets
df = pd.read_csv("hiv_data.csv")
df_test = pd.read_csv("hiv_test_data.csv")

# Standardize column names (strip spaces)
df.columns = df.columns.str.strip()
df_test.columns = df_test.columns.str.strip()

# Ensure consistency in column order between training and test data
df_test = df_test[df.columns.intersection(df_test.columns)]

# Separate features and labels (ignoring Country column)
X = df.drop(columns=["Country", "CountryID", "Risk Level"], errors="ignore")
y = df["Risk Level"].map({"High Risk": 1, "Low Risk": 0})

# Store country names, country IDs, and correctly mapped actual risk levels
test_countries = df_test["Country"].tolist()
test_country_ids = df_test["CountryID"].tolist()
actual_risk_levels = df_test["Risk Level"].map({"High Risk": 1, "Low Risk": 0}).tolist()

# Drop Country, CountryID, and Risk Level from test data
X_test = df_test.drop(columns=["Country", "CountryID", "Risk Level"], errors="ignore")

# Convert all data to numeric
X = X.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')

# Handle missing values (Fill NaNs with column means)
X.fillna(X.mean(), inplace=True)
X_test.fillna(X.mean(), inplace=True)  # Use training set mean to avoid data leakage

# Normalize (min-max scaling)
X_min, X_max = X.min(), X.max()
X = (X - X_min) / (X_max - X_min)
X_test = (X_test - X_min) / (X_max - X_min)  # Use same scaling as training set

# Convert to NumPy arrays
X_np = X.to_numpy()
y_np = y.to_numpy()
X_test_np = X_test.to_numpy()

# ==============================
# 2. DISTANCE & SIMILARITY METRICS
# ==============================

# ---- EUCLIDEAN DISTANCE ----
def euclidean_distance(X_train, X_test):
    # Computes Euclidean distance between each row in X_train and X_test.
    return np.linalg.norm(X_train - X_test, axis=1)

# ---- COSINE SIMILARITY ----
def cosine_similarity(X_train, X_test):
    # Computes cosine similarity between X_train and X_test (higher means more similar).
    return np.dot(X_train, X_test) / (np.linalg.norm(X_train, axis=1) * np.linalg.norm(X_test))

# ---- MAHALANOBIS DISTANCE ----
def mahalanobis_distance(X_train, X_test):
    # Computes Mahalanobis distance to measure how different test points are from the mean.
    cov_matrix = np.cov(X_train.T)
    inv_cov_matrix = np.linalg.pinv(cov_matrix)  # Use pseudo-inverse for stability
    means = np.mean(X_train, axis=0)

    distances = []
    for test_point in X_test:
        diff = test_point - means
        dist = np.sqrt(np.dot(np.dot(diff, inv_cov_matrix), diff.T))
        distances.append(dist)

    return np.array(distances)

# ==============================
# 3. K-NEAREST NEIGHBORS (KNN) CLASSIFIER
# ==============================

# Prompt user to enter the value of k
k = int(input("Enter the number of nearest neighbors (k) for KNN: "))

# ---- KNN CLASSIFIER USING EUCLIDEAN DISTANCE ----
def knn_euclidean(X_train, y_train, X_test):
    predictions = []
    for test_point in X_test:
        distances = euclidean_distance(X_train, test_point)
        nearest_neighbors = np.argsort(distances)[:k] # Select k nearest neighbors
        prediction = np.round(np.mean(y_train[nearest_neighbors])) # Majority vote
        predictions.append(int(prediction))
    return np.array(predictions)

# ---- KNN CLASSIFIER USING COSINE SIMILARITY ----
def knn_cosine(X_train, y_train, X_test):
    predictions = []
    for test_point in X_test:
        similarities = cosine_similarity(X_train, test_point)
        nearest_neighbors = np.argsort(-similarities)[:k]  # Higher similarity = closer neighbor
        prediction = np.round(np.mean(y_train[nearest_neighbors])) 
        predictions.append(int(prediction))
    return np.array(predictions)

# ---- KNN CLASSIFIER USING MAHALANOBIS DISTANCE ----
def knn_mahalanobis(X_train, y_train, X_test):
    distances = mahalanobis_distance(X_train, X_test)
    threshold = np.percentile(mahalanobis_distance(X_train, X_train), 60)  # Adjusted percentile for better balance
    return np.where(distances > threshold, 1, 0)  # High distance = High Risk

# ==============================
# 4. CROSS VALIDATION
# ==============================

def cross_validation(X, y, folds=5):
    # Performs k-fold cross-validation to evaluate model accuracy and precision
    fold_size = len(X) // folds 
    accuracies = []
    precisions = []

    for i in range(folds):
        # Split dataset into training and validation folds
        X_test_fold = X[i * fold_size:(i + 1) * fold_size]
        y_test_fold = y[i * fold_size:(i + 1) * fold_size]
        X_train_fold = np.concatenate((X[:i * fold_size], X[(i + 1) * fold_size:]), axis=0)
        y_train_fold = np.concatenate((y[:i * fold_size], y[(i + 1) * fold_size:]), axis=0)

        # KNN (Euclidean)
        y_pred_knn_euclidean = knn_euclidean(X_train_fold, y_train_fold, X_test_fold)
        accuracy_knn_euclidean = np.mean(y_pred_knn_euclidean == y_test_fold)
        precision_knn_euclidean = np.sum((y_pred_knn_euclidean == 1) & (y_test_fold == 1)) / max(np.sum(y_pred_knn_euclidean == 1), 1)

        # KNN (Cosine)
        y_pred_knn_cosine = knn_cosine(X_train_fold, y_train_fold, X_test_fold)
        accuracy_knn_cosine = np.mean(y_pred_knn_cosine == y_test_fold)
        precision_knn_cosine = np.sum((y_pred_knn_cosine == 1) & (y_test_fold == 1)) / max(np.sum(y_pred_knn_cosine == 1), 1)

        # Mahalanobis
        mahal_dist = mahalanobis_distance(X_train_fold, X_test_fold)
        threshold = np.percentile(mahalanobis_distance(X_train_fold, X_train_fold), 75)
        y_pred_mahal = np.where(mahal_dist > threshold, 1, 0)  
        accuracy_mahal = np.mean(y_pred_mahal == y_test_fold)
        precision_mahal = np.sum((y_pred_mahal == 1) & (y_test_fold == 1)) / max(np.sum(y_pred_mahal == 1), 1)

        accuracies.append((accuracy_knn_euclidean, accuracy_knn_cosine, accuracy_mahal))
        precisions.append((precision_knn_euclidean, precision_knn_cosine, precision_mahal))

    return np.mean(accuracies, axis=0), np.mean(precisions, axis=0)

# Perform cross-validation
(acc_euclidean, acc_cosine, acc_mahal), (prec_euclidean, prec_cosine, prec_mahal) = cross_validation(X_np, y_np)

# ==============================
# 5. CLASSIFY TEST COUNTRIES & COMPUTE ACCURACY/PRECISION
# ==============================

# Predict risk levels for test data using KNN (Euclidean)
y_test_pred_euclidean = knn_euclidean(X_np, y_np, X_test_np)

# Predict risk levels for test data using KNN (Cosine Similarity)
y_test_pred_cosine = knn_cosine(X_np, y_np, X_test_np)

# Predict risk levels for test data using Mahalanobis Distance
# If the Mahalanobis distance is above the median training distance, classify as High Risk (1), otherwise Low Risk (0)
y_test_pred_mahal = np.where(mahalanobis_distance(X_np, X_test_np) > np.median(mahalanobis_distance(X_np, X_np)), 1, 0)

# Convert actual risk levels to NumPy array
actual_risk_levels_np = np.array(actual_risk_levels, dtype=int)

# ---- COMPUTE MODEL ACCURACY ----
# Accuracy measures how often the model's predictions match the actual risk levels in test data
accuracy_euclidean = np.mean(y_test_pred_euclidean == actual_risk_levels_np)
accuracy_cosine = np.mean(y_test_pred_cosine == actual_risk_levels_np)
accuracy_mahal = np.mean(y_test_pred_mahal == actual_risk_levels_np)

# ---- COMPUTE MODEL PRECISION ----
# Precision measures how many of the predicted "High Risk" classifications were actually correct
precision_euclidean = np.sum((y_test_pred_euclidean == 1) & (actual_risk_levels_np == 1)) / max(np.sum(y_test_pred_euclidean == 1), 1)
precision_cosine = np.sum((y_test_pred_cosine == 1) & (actual_risk_levels_np == 1)) / max(np.sum(y_test_pred_cosine == 1), 1)
precision_mahal = np.sum((y_test_pred_mahal == 1) & (actual_risk_levels_np == 1)) / max(np.sum(y_test_pred_mahal == 1), 1)


# ==============================
# 6. PRINT FORMATTED RESULTS
# ==============================

print(f"\nKNN CLASSIFIER with (k) = {k}")
print("\nCross-Validation Accuracies:")
print(f"KNN (Euclidean): {acc_euclidean:.2f}")
print(f"KNN (Cosine Similarity): {acc_cosine:.2f}")
print(f"Mahalanobis Distance: {acc_mahal:.2f}")

print("\nCross-Validation Precision:")
print(f"KNN (Euclidean): {prec_euclidean:.2f}")
print(f"KNN (Cosine Similarity): {prec_cosine:.2f}")
print(f"Mahalanobis Distance: {prec_mahal:.2f}")

print("\nModel Accuracy on Test Data:")
print(f"KNN (Euclidean): {accuracy_euclidean:.2f}")
print(f"KNN (Cosine Similarity): {accuracy_cosine:.2f}")
print(f"Mahalanobis Distance: {accuracy_mahal:.2f}")

print("\nModel Precision on Test Data:")
print(f"KNN (Euclidean): {precision_euclidean:.2f}")
print(f"KNN (Cosine Similarity): {precision_cosine:.2f}")
print(f"Mahalanobis Distance: {precision_mahal:.2f}")

# Print predictions for 47 countries
print("\nID    Country                   Euclidean  Cosine     Mahalanobis Actual")
print("============================================================================")
for i, (country_name, country_id) in enumerate(zip(test_countries, test_country_ids)):
    risk_euclidean = "High" if y_test_pred_euclidean[i] == 1 else "Low"
    risk_cosine = "High" if y_test_pred_cosine[i] == 1 else "Low"
    risk_mahal = "High" if y_test_pred_mahal[i] == 1 else "Low"
    actual_risk = "High" if actual_risk_levels[i] == 1 else "Low"
    
    print(f"{country_id:<5} {country_name:<25} {risk_euclidean:<10} {risk_cosine:<10} {risk_mahal:<10} {actual_risk:<10}")
