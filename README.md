# Iris Classifier

This repository contains an implementation of an Iris flower classifier using multiple machine learning algorithms. The model is trained on the famous Iris dataset, which consists of three different species: Setosa, Versicolor, and Virginica. The classifier predicts the species based on four features: sepal length, sepal width, petal length, and petal width.

## Live Demo
You can check the real-time predictions of the classifier here:
[https://iris-classifier-by-armanlaliwala.onrender.com](https://iris-classifier-by-armanlaliwala.onrender.com)

## Project Overview
The project involves multiple machine learning algorithms to classify the Iris dataset. The algorithms used include:
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Decision Tree**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**

Additionally, a manual model based on `if-else` conditions is implemented to compare accuracy with machine learning models.

## Key Observations
1. **Accuracy of Machine Learning Models:**
   - All the machine learning models yield nearly the same accuracy on the Iris dataset due to its small size and well-separated classes.
   
2. **Manual Model vs. Machine Learning Models:**
   - The manually designed model using `if-else` conditions has significantly lower accuracy compared to machine learning models.
   - This highlights the advantage of data-driven approaches over rule-based methods in classification tasks.
   
3. **Effect of Fine-Tuning:**
   - After hyperparameter tuning, the accuracy of some models improves slightly.
   - Techniques such as adjusting `C` in SVM, optimizing `n_neighbors` in KNN, and pruning the Decision Tree help in refining predictions.

## Code Explanation
### 1. Data Preprocessing
- Load the Iris dataset using `sklearn.datasets.load_iris()`.
- Split the dataset into training and testing sets using `train_test_split()`.
- Standardize the data where necessary to optimize model performance.

### 2. Model Training
- Train different classifiers using the `fit()` method on training data.
- Evaluate models using accuracy scores on the test dataset.

### 3. Manual Model (If-Else Conditions)
- Implemented a simple `if-else` logic to classify data.
- Compared its performance with ML models.

### 4. Model Evaluation
- Calculated accuracy scores for all models.
- Displayed comparative results showing the advantage of ML models over a rule-based approach.

## Results
| Model                 | Accuracy |
|----------------------|----------|
| Logistic Regression | ~97%     |
| SVM                 | ~97%     |
| Decision Tree       | ~97%     |
| Random Forest       | ~97%     |
| KNN                 | ~97%     |
| **Manual Model**    | ~70%     |

### Conclusion
- The Iris dataset is simple and well-structured, which is why all ML models perform similarly.
- Fine-tuning helps in optimizing results but does not significantly change accuracy.
- The manual model struggles due to the complexity of class boundaries.

## How to Run the Project Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/Armanlaliwala/iris-classifier
   cd iris-classifier
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the classifier script:
   ```bash
   streamlit run streamlit.py
   ```

## Future Enhancements
- Implementing deep learning models for more complex classification tasks.
- Enhancing the web interface for a better user experience.
- Exploring advanced feature engineering techniques to further improve accuracy.

---
Developed by **Arman Laliwala**

