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

def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

def cosine_similarity_distance(x, y):
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def mahalanobis_distance(x, y, X_train):
    cov_matrix = np.cov(X_train.T)
    inv_cov_matrix = np.linalg.pinv(cov_matrix)
    diff = x - y
    return np.sqrt(np.dot(np.dot(diff, inv_cov_matrix), diff.T))

# ==============================
# 3. CENTROID BASED CLASSIFIER
# ==============================

# ---- CENTROID CALCULATION ----
def compute_centroid(X_train, y_train, label):
    return np.mean(X_train[y_train == label], axis=0)

# Compute centroids for high and low risk groups
centroid_high = compute_centroid(X_np, y_np, 1)
centroid_low = compute_centroid(X_np, y_np, 0)

# ==============================
# 4. CROSS VALIDATION
# ==============================

def cross_validation_centroid(X, y, folds=5):
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
        
        centroid_high = compute_centroid(X_train_fold, y_train_fold, 1)
        centroid_low = compute_centroid(X_train_fold, y_train_fold, 0)
        
        # CENTROID Euclidean
        y_pred_euclidean = classify_by_centroid(X_test_fold, euclidean_distance, centroid_high, centroid_low)
        accuracy_euclidean = np.mean(y_pred_euclidean == y_test_fold)
        precision_euclidean = np.sum((y_pred_euclidean == 1) & (y_test_fold == 1)) / max(np.sum(y_pred_euclidean == 1), 1)
        
        # CENTROID Cosine
        y_pred_cosine = classify_by_centroid(X_test_fold, cosine_similarity_distance, centroid_high, centroid_low)
        accuracy_cosine = np.mean(y_pred_cosine == y_test_fold)
        precision_cosine = np.sum((y_pred_cosine == 1) & (y_test_fold == 1)) / max(np.sum(y_pred_cosine == 1), 1)
        
        # CENTROID Mahalanobis
        y_pred_mahal = classify_by_centroid(X_test_fold, lambda x, y: mahalanobis_distance(x, y, X_train_fold), centroid_high, centroid_low)
        accuracy_mahal = np.mean(y_pred_mahal == y_test_fold)
        precision_mahal = np.sum((y_pred_mahal == 1) & (y_test_fold == 1)) / max(np.sum(y_pred_mahal == 1), 1)
        
        accuracies.append((accuracy_euclidean, accuracy_cosine, accuracy_mahal))
        precisions.append((precision_euclidean, precision_cosine, precision_mahal))

    return np.mean(accuracies, axis=0), np.mean(precisions, axis=0)

# Perform cross-validation
(acc_euclidean, acc_cosine, acc_mahal), (prec_euclidean, prec_cosine, prec_mahal) = cross_validation_centroid(X_np, y_np)

# ==============================
# 5. CLASSIFY TEST COUNTRIES & COMPUTE ACCURACY/PRECISION
# ==============================

def classify_by_centroid(X_test, metric_function, centroid_high, centroid_low):
    predictions = []
    for test_point in X_test:
        dist_high = metric_function(test_point, centroid_high)
        dist_low = metric_function(test_point, centroid_low)
        prediction = 1 if dist_high < dist_low else 0  # Assign label based on closest centroid
        predictions.append(prediction)
    return np.array(predictions)


# Compute distances using the three metrics
y_test_pred_euclidean = classify_by_centroid(X_test_np, euclidean_distance, centroid_high, centroid_low)
y_test_pred_cosine = classify_by_centroid(X_test_np, cosine_similarity_distance, centroid_high, centroid_low)
y_test_pred_mahal = classify_by_centroid(X_test_np, lambda x, y: mahalanobis_distance(x, y, X_np), centroid_high, centroid_low)

# Convert actual risk levels to a NumPy array for easy comparison
actual_risk_levels_np = np.array(actual_risk_levels, dtype=int)

# ---- MODEL ACCURACY ON TEST DATA ----
# Accuracy is calculated as the proportion of correctly classified instances
accuracy_euclidean = np.mean(y_test_pred_euclidean == actual_risk_levels_np)
accuracy_cosine = np.mean(y_test_pred_cosine == actual_risk_levels_np)
accuracy_mahal = np.mean(y_test_pred_mahal == actual_risk_levels_np)

# ---- MODEL PRECISION ON TEST DATA ----
# Precision measures how many of the predicted High Risk cases are actually High Risk
precision_euclidean = np.sum((y_test_pred_euclidean == 1) & (actual_risk_levels_np == 1)) / max(np.sum(y_test_pred_euclidean == 1), 1)
precision_cosine = np.sum((y_test_pred_cosine == 1) & (actual_risk_levels_np == 1)) / max(np.sum(y_test_pred_cosine == 1), 1)
precision_mahal = np.sum((y_test_pred_mahal == 1) & (actual_risk_levels_np == 1)) / max(np.sum(y_test_pred_mahal == 1), 1)

# ==============================
# 6. PRINT FORMATTED RESULTS
# ==============================

print("\nCENTROID-BASED CLASSIFIER")
print("\nCross-Validation Accuracies:")
print(f"Euclidean: {acc_euclidean:.2f}")
print(f"Cosine Similarity: {acc_cosine:.2f}")
print(f"Mahalanobis: {acc_mahal:.2f}")

print("\nCross-Validation Precision:")
print(f"Euclidean: {prec_euclidean:.2f}")
print(f"Cosine Similarity: {prec_cosine:.2f}")
print(f"Mahalanobis: {prec_mahal:.2f}")

print("\nModel Accuracy on Test Data:")
print(f"KNN (Euclidean): {accuracy_euclidean:.2f}")
print(f"KNN (Cosine Similarity): {accuracy_cosine:.2f}")
print(f"Mahalanobis Classifier: {accuracy_mahal:.2f}")

print("\nModel Precision on Test Data:")
print(f"KNN (Euclidean): {precision_euclidean:.2f}")
print(f"KNN (Cosine Similarity): {precision_cosine:.2f}")
print(f"Mahalanobis Classifier: {precision_mahal:.2f}")

# Print predictions for 47 countries
print("\nID    Country                   Euclidean  Cosine     Mahalanobis Actual")
print("============================================================================")
for i, (country_name, country_id) in enumerate(zip(test_countries, test_country_ids)):
    risk_euclidean = "High" if y_test_pred_euclidean[i] == 1 else "Low"
    risk_cosine = "High" if y_test_pred_cosine[i] == 1 else "Low"
    risk_mahal = "High" if y_test_pred_mahal[i] == 1 else "Low"
    actual_risk = "High" if actual_risk_levels[i] == 1 else "Low"
    
    print(f"{country_id:<5} {country_name:<25} {risk_euclidean:<10} {risk_cosine:<10} {risk_mahal:<10} {actual_risk:<10}")