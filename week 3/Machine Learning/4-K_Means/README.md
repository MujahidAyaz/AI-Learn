# K-Means Clustering — Customer Segmentation

An unsupervised learning project that discovers natural customer segments from mall customer data using income and spending behavior.

## Project Overview

This project uses K-Means clustering to group customers without any labeled target column. Unlike supervised learning, the goal here is pattern discovery, not predicting a known answer.

The script walks through data loading, feature scaling, choosing the best number of clusters, training the final model, interpreting each segment, visualizing the results, and assigning a new customer to a cluster.

## What the Code Does

- Loads the `mall_customers.csv` dataset.
- Selects `Annual Income (k$)` and `Spending Score (1-100)` as clustering features.
- Scales the features with `StandardScaler` because K-Means depends on distance.
- Tests K values from 2 to 10 using the Elbow Method and Silhouette Score.
- Trains the final K-Means model with the best K.
- Summarizes each cluster and gives it a business-friendly name.
- Plots the final customer segments and centroids.
- Predicts the cluster for a new customer with income 75k and spending score 85.

## Why Scaling Matters

K-Means works by measuring distances between points. If one feature has much larger numeric values than another, it can dominate the clustering process and distort the result. Scaling makes both features contribute fairly.

## Cluster Interpretation

The code turns cluster numbers into practical business segments such as average customers, VIPs, impulsive spenders, careful high earners, and budget shoppers. That makes the results easier to understand and use for marketing decisions.

## Files Used

| File | Purpose |
|---|---|
| `mall_customers.csv` | Input dataset containing customer income and spending score |
| `kmeans_elbow.png` | Elbow and silhouette plots for selecting K |
| `kmeans_final_clusters.png` | Final scatter plot of clustered customers |

## How to Run

```bash
pip install pandas numpy matplotlib scikit-learn
python your_script.py
```

## Expected Output

When you run the script, it prints dataset information, scaling results, the best K based on silhouette score, cluster counts, cluster averages, and the predicted segment for a new customer. It also saves the plots to image files for review.

## Learning Focus

This tutorial is useful for understanding:

- Unsupervised learning.
- Distance-based clustering.
- The difference between inertia and silhouette score.
- How to interpret clusters in a business context.
- How to use a clustering model for a new data point.

## Notes

The exact best K and cluster statistics can vary slightly depending on the dataset version and random initialization, but the workflow remains the same. The code uses `random_state=42` and `n_init=10` to make the results reproducible.