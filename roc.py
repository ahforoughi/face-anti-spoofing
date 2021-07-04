from sklearn.metrics import roc_curve                # Calculate the ROC curve
import pandas as pd 

results = pd.read_csv('result.csv')
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
