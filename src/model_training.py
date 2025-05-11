# src/model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline # Import Pipeline


def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        target_column (str): The name of the target variable column.
        test_size (float): The proportion of the data to use for the test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y) # Add stratify=y
    print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples) sets.")
    return X_train, X_test, y_train, y_test



def train_model(X_train, y_train, model_name='logistic_regression', params=None, cv_method='holdout'):
    """
    Trains a machine learning model.

    Args:
        X_train (pandas.DataFrame): The training features.
        y_train (pandas.Series): The training target variable.
        model_name (str): The name of the model to train
            ('logistic_regression', 'random_forest', 'naive_bayes', 'svm', 'decision_tree',
             'knn', 'neural_network', 'gradient_boosting', 'adaboost', 'lda', 'qda',
             'gaussian_process', 'calibrated_cv', 'voting_classifier').
        params (dict, optional): A dictionary of model parameters. If None, default
            parameters are used.
        cv_method (str): The cross-validation method ('holdout', 'kfold', 'stratifiedkfold').
            If 'holdout', no cross-validation is performed.

    Returns:
        tuple: (model, cv_results)
            model: The trained model.
            cv_results:  Cross validation results (if applicable). If holdout, returns None
    """
    model = None
    cv_results = None
    model_params = params or {} # Use this to avoid overwriting

    if model_name == 'logistic_regression':
        #  Pipeline:  Imputation, Scaling, and Logistic Regression
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),  # Scale the data
            ('logistic_regression', LogisticRegression(random_state=42,  **model_params))  # Add C for regularization
        ])
        model = pipeline
    elif model_name == 'random_forest':
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values
            ('random_forest', RandomForestClassifier(random_state=42, **model_params))
        ])
        model = pipeline
    elif model_name == 'naive_bayes':
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('naive_bayes', GaussianNB(**model_params))
        ])
        model = pipeline
    elif model_name == 'svm':
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),  # Scale the data
            ('svm', SVC(random_state=42, probability=True, **model_params))  # probability=True for ROC AUC
        ])
        model = pipeline
    elif model_name == 'decision_tree':
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('decision_tree', DecisionTreeClassifier(random_state=42, **model_params))
        ])
        model = pipeline
    elif model_name == 'knn':
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),  # Scale the data
            ('knn', KNeighborsClassifier(**model_params))
        ])
        model = pipeline
    elif model_name == 'neural_network':
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),  # Scale the data
            ('mlp', MLPClassifier(random_state=42, **model_params))
        ])
        model = pipeline
    elif model_name == 'gradient_boosting':
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('gradient_boosting', GradientBoostingClassifier(random_state=42, **model_params))
        ])
        model = pipeline
    elif model_name == 'adaboost':
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('adaboost', AdaBoostClassifier(random_state=42, **model_params))
        ])
        model = pipeline
    elif model_name == 'lda':
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),  # Scale the data
            ('lda', LinearDiscriminantAnalysis(**model_params))
        ])
        model = pipeline
    elif model_name == 'qda':
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),  # Scale the data
            ('qda', QuadraticDiscriminantAnalysis(**model_params))
        ])
        model = pipeline
    elif model_name == 'gaussian_process':
        kernel = RBF(length_scale=1.0)  # Example kernel, can be part of params
        model = GaussianProcessClassifier(kernel=kernel, random_state=42, **model_params)
    elif model_name == 'calibrated_cv':
        base_model = GaussianNB()  # Or any other classifier
        model = CalibratedClassifierCV(base_model, cv=5, method='isotonic')
    elif model_name == 'voting_classifier':
        # Example of setting up a voting classifier.  You would need to
        # define the estimators in the params.
        if not params or 'estimators' not in params:
            raise ValueError("For voting_classifier, the 'estimators' parameter must be provided in params.")
        model = VotingClassifier(**model_params)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    print(f"Training model: {model_name}")
    if cv_method == 'holdout':
        model.fit(X_train, y_train) # Fit on the original training data
    else:
        if cv_method == 'kfold':
            cv = KFold(n_splits=min(5, len(X_train)), shuffle=True, random_state=42)  # Use a maximum of 5 splits or the number of training samples
        elif cv_method == 'stratifiedkfold':
            cv = StratifiedKFold(n_splits=min(3, len(X_train)), shuffle=True, random_state=42) # Use a maximum of 3 splits or the number of samples in the smallest class
        else:
            raise ValueError(f"Invalid cross-validation method: {cv_method}")
        
        cv_results = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc') # Use a meaningful score, on unscaled, unimputed data
        print(f"Cross-validation ({cv_method}) results (ROC AUC): {cv_results.mean()} +/- {cv_results.std()}")
        model.fit(X_train, y_train) # Fit on the FULL training set AFTER cross validation, on unscaled, unimputed data
    return model, cv_results



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

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of the positive class

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
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'target': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    })
    
    # Add some missing values for demonstration
    data.loc[2, 'feature1'] = None
    data.loc[5, 'feature2'] = None
    
    print("Original Data:")
    print(data)
    print("\nTarget Variable Distribution:")
    print(data['target'].value_counts())  # Print class distribution

    X_train, X_test, y_train, y_test = split_data(data, 'target', test_size=0.3, random_state=42)

    # Example 1: Logistic Regression
    lr_model, lr_cv_results = train_model(X_train, y_train, model_name='logistic_regression',
                                     params={'C': 0.1}, cv_method='stratifiedkfold') #changed to stratified kfold
    if lr_cv_results is not None:
        print(f"Logistic Regression CV Results: {lr_cv_results}")
    lr_evaluation = evaluate_model(lr_model, X_test, y_test, model_name='Logistic Regression')

    # Example 2: Random Forest
    rf_model, rf_cv_results = train_model(X_train, y_train, model_name='random_forest',
                                     params={'n_estimators': 100, 'max_depth': 5}, cv_method='stratifiedkfold')
    if rf_cv_results is not None:
        print(f"Random Forest CV Results: {rf_cv_results}")
    rf_evaluation = evaluate_model(rf_model, X_test, y_test, model_name='Random Forest')
    
    # Example 3: Gaussian Naive Bayes (no params)
    nb_model, nb_cv_results = train_model(X_train, y_train, model_name='naive_bayes', cv_method='holdout')
    if nb_cv_results is not None:
        print(f"Naive Bayes CV Results: {nb_cv_results}")
    nb_evaluation = evaluate_model(nb_model, X_test, y_test, model_name='Naive Bayes')

