# Hierarchical Clustering - Mall Customer Segmentation

## Overview

This project demonstrates Agglomerative Hierarchical Clustering using the Mall Customers dataset.

The objective is to group customers into meaningful segments based on:

- Annual Income
- Spending Score

Unlike K-Means, Hierarchical Clustering does not require specifying the number of clusters before building the hierarchy. A dendrogram is used to determine the optimal number of clusters.

---

## Dataset

Mall Customers Dataset

Features:

- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

Only the following features are used for clustering:

- Annual Income
- Spending Score

---

## Technologies

- Python
- Pandas
- Matplotlib
- Scikit-Learn
- SciPy

---

## Project Workflow

1. Load dataset from URL
2. Explore dataset
3. Check missing values
4. Select relevant features
5. Build dendrogram
6. Determine optimal clusters
7. Train Agglomerative Hierarchical Clustering model
8. Visualize customer groups
9. Analyze cluster statistics

---

## Algorithm

Agglomerative Hierarchical Clustering

- Linkage: Ward
- Distance Metric: Euclidean
- Number of Clusters: 5

---

## Results

The model successfully segmented customers into five different spending behavior groups.

These clusters can help businesses:

- Improve targeted marketing
- Understand customer behavior
- Design personalized promotions
- Increase customer retention

---

## Learning Outcomes

Through this project you will understand:

- Hierarchical Clustering
- Agglomerative Clustering
- Ward Linkage
- Euclidean Distance
- Dendrogram Interpretation
- Customer Segmentation
- Cluster Visualization

---

## Future Improvements

- Standardize features before clustering.
- Compare different linkage methods (single, complete, average, ward).
- Evaluate cluster quality using metrics like the Silhouette Score.
- Experiment with other clustering algorithms such as DBSCAN.
