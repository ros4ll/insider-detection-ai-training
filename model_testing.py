import joblib
import sys
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

# Select model to evaluate from arguments
parser = argparse.ArgumentParser(description="Select the model to evaluate")
parser.add_argument("model",type=str, help=" Select from sgd, svm")

args = parser.parse_args()
model = args.model
if model is not None:
    print('Evaluating model:', model)
else:
    print('Error: model selection is required.')
    sys.exit(1)
# load global model
global_model_path = f"/tmp/nvflare/{model}/workspace/simulate_job/app_server/model_param.joblib"
model_dict = joblib.load(global_model_path)
#load testing dataset 
test_data = pd.read_csv("data/site-1/test_data.csv")
test_data = test_data.sample(n=10000)
X_test = test_data.drop(columns=["label"])
y_test=test_data["label"]
if model == "svm":
    # Extract support vectors
    support_x = model_dict['support_x']
    support_y = model_dict['support_y']

    # Construct an SVM model
    global_model = SVC(kernel='rbf',gamma="scale", class_weight="balanced")
    global_model.fit(X=support_x, y=support_y)
    predictions = global_model.predict(X_test)

elif model == "sgd":
    # Extract weights
    coef = model_dict['coef']
    intercept = model_dict['intercept']
    global_model = SGDClassifier(loss="squared_hinge",penalty="l2",fit_intercept= 1,learning_rate= "optimal",eta0= 1e-3, max_iter=1)
    global_model.coef_=coef
    global_model.intercept_=intercept
    # evaluate the model
    decision_scores = global_model.decision_function(X_test)
    # Make predictions based on the sign of the raw scores
    predictions = (decision_scores > 0).astype(int)
else: 
    print('Error: model selection is not valid. Select from sgd, svm')
    sys.exit(1)




# metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='macro', zero_division=1)
recall = recall_score(y_test, predictions, average='macro')
f1 = f1_score(y_test, predictions, average = 'macro')
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
