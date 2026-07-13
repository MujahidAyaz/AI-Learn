"""
Training Pipeline
A professional training loop looks like this:

Forward Pass
      ↓
Compute Loss
      ↓
Backpropagation
      ↓
Update Weights
      ↓
Evaluate Metrics

Notice:
Metrics never update the weights.
Loss does.
"""

# NumPy Implementation
import numpy as np

y_true = np.array([1,0,1,1,0,1,0,0])

y_pred = np.array([1,0,0,1,0,1,1,0])

tp = np.sum((y_true==1) & (y_pred==1))

tn = np.sum((y_true==0) & (y_pred==0))

fp = np.sum((y_true==0) & (y_pred==1))

fn = np.sum((y_true==1) & (y_pred==0))

accuracy = (tp+tn)/len(y_true)

precision = tp/(tp+fp)

recall = tp/(tp+fn)

f1 = 2*precision*recall/(precision+recall)

print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1 Score : {f1:.3f}")