# Titanic Survival Prediction

## Overview
This project builds and end-to-end machine learning pipeline to predict passenger survival on the Titanic dataset.

## Workflow
- Data exploration
- Train/test split
- Missing value handling
- Numerical and categorical preprocessing
- Logistic Regression and Random Forest comparison
- Cross-validation
- Hyperparameter tuning with GridSearchCV
- Final test evaluation
- Error analysis

## Models
- Logistic Regression
- Random Forest Classifier

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score

## Final Results 
- Selected model: Logistic Regression
- Best hyperparameters: 0.1
- Best CV F1: 0.704
- Final accuracy: 0.813
- Final precision: 0.774
- Final recall: 0.72
- Final F1: 0.746
- CV vs test difference: 0.042

## Key Finding
Many false negatives were male survivors, while many false positives were female non-survivors, suggesting that the model may rely heavily on sex when making predictions.

## API

The trained model is exposed through a FastAPI application.

### Endpoints
- 'Get /health' - Check API status
- 'POST /predict' - Generate a survival prediction

### Example Input
{
  "pclass": 1,
  "sex": "female",
  "age": 30,
  "sibsp": 0,
  "parch": 0,
  "fare": 80,
  "embarked": "S"
}
