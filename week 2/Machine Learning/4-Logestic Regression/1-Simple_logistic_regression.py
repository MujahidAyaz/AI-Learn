from sklearn.linear_model import LogisticRegression

# Each row = one person
# [salary, number of existing loans]
X = [
    [5000, 0],   # person 1
    [3000, 1],   # person 2
    [1000, 3],   # person 3
    [8000, 0],   # person 4
]

# 1 = repaid, 0 = didn't repay
y = [1, 1, 0, 1]

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Predict a new person: earns 4000, has 1 loan
print(model.predict([[4000, 1]]))         # → [1] means YES
print(model.predict_proba([[4000, 1]]))   # → probability of yes/no