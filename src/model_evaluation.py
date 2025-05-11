# src/model_evaluation.py
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer # Import SimpleImputer


def evaluate_model(model, X_test, y_test, model_name=""):
    """
    Evaluates the trained machine learning model.

    Args:
        model: The trained model.
        X_test (pandas.DataFrame): The testing features.
        y_test (pandas.Series): The testing target variable.
        model_name (str):  Name of the model (for reporting)

    Returns:
        dict: A dictionary of evaluation metrics.
    """

    # Impute missing values in test set using the *same* imputer fitted on the training data
    imputer = SimpleImputer(strategy='mean')
    X_test_imputed = imputer.transform(X_test)
    X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns)

    y_pred = model.predict(X_test_imputed)
    y_pred_proba = model.predict_proba(X_test_imputed)[:, 1]  # Probability of the positive class

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Evaluation Metrics for {model_name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} ROC AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }



if __name__ == "__main__":
    # Example Usage
    # Create a sample dataset (replace with your actual data)
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'target': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    })

    #  Add some missing values
    data.loc[2, 'feature1'] = None
    data.loc[5, 'feature2'] = None
    

    # Create a dummy model for demonstration
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()

    # Split the data
    X = data.drop('target', axis=1)
    y = data['target']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the dummy model (replace with your actual trained model)
    model.fit(X_train, y_train) #  You'd replace this with your actual training

    # Evaluate the model
    evaluation_metrics = evaluate_model(model, X_test, y_test, model_name='Dummy Logistic Regression')
    print(evaluation_metrics)
