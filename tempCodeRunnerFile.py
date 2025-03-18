
# y_test_pred_euclidean = classify_by_centroid(X_test_np, euclidean_distance, centroid_high, centroid_low)
# y_test_pred_cosine = classify_by_centroid(X_test_np, cosine_similarity, centroid_high, centroid_low)
# y_test_pred_mahal = classify_by_centroid(X_test_np, lambda x, y: mahalanobis_distance(x, y, X_np), centroid_high, centroid_low)

# # Print formatted predictions for 47 countries
# print("\nID    Country                   Euclidean  Cosine     Mahalanobis Actual")
# print("============================================================================")
# for i, (country_name, country_id) in enumerate(zip(test_countries, test_country_ids)):
#     risk_euclidean = "High" if y_test_pred_euclidean[i] == 1 else "Low"
#     risk_cosine = "High" if y_test_pred_cosine[i] == 1 else "Low"
#     risk_mahal = "High" if y_test_pred_mahal[i] == 1 else "Low"
#     actual_risk = "High" if actual_risk_levels[i] == 1 else "Low"
    
#     print(f"{country_id:<5} {country_name:<25} {risk_euclidean:<10} {risk_cosine:<10} {risk_mahal:<10} {actual_risk:<10}")