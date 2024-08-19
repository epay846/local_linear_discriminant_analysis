import numpy as np

def gaussian_kernel(x0, xi, lambda_):
    return np.exp(-np.linalg.norm(x0 - xi)**2 / (2 * lambda_**2))
def weighted_stats(X, y, x0, lambda_):
    classes = np.unique(y)
    means = {}
    covariances = {}

    for cls in classes:
        X_cls = X[y == cls]
        weights = np.array([gaussian_kernel(x0, xi, lambda_) for xi in X_cls])
        weights_sum = np.sum(weights)
        
        if weights_sum == 0:
            weights = np.ones_like(weights) / len(weights)
            weights_sum = np.sum(weights)
        
        weights /= weights_sum
        
        mean = np.average(X_cls, axis=0, weights=weights)
        centered = X_cls - mean
        cov = np.dot(weights * centered.T, centered) / np.sum(weights)
        
        means[cls] = mean
        covariances[cls] = cov
    
    return means, covariances
  
def lda_decision(x, means, covariances, priors):
    classes = list(means.keys())
    scores = []
    
    for cls in classes:
        mean = means[cls]
        cov = covariances[cls]
        prior = priors[cls]
        
        inv_cov = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)  
        score = -0.5 * np.dot(np.dot((x - mean).T, inv_cov), (x - mean)) + np.log(prior)
        scores.append(score)
    
    return classes[np.argmax(scores)]


from sklearn.metrics import accuracy_score

def local_lda(X_train, y_train, X_test, y_test, lambdas):
    priors = {cls: np.mean(y_train == cls) for cls in np.unique(y_train)}
    training_errors = []
    test_errors = []
    
    for lambda_ in lambdas:
        print(f"Processing lambda = {lambda_}")
        y_train_pred = []
        for i in range(len(X_train)):
            if i % 100 == 0:  
                print(f"Processing training sample {i}/{len(X_train)}")
            x0 = X_train[i]
            means, covariances = weighted_stats(X_train, y_train, x0, lambda_)
            y_pred = lda_decision(x0, means, covariances, priors)
            y_train_pred.append(y_pred)
        
        y_test_pred = []
        for i, x0 in enumerate(X_test):
            if i % 100 == 0:  
                print(f"Processing test sample {i}/{len(X_test)}")
            means, covariances = weighted_stats(X_train, y_train, x0, lambda_)
            y_pred = lda_decision(x0, means, covariances, priors)
            y_test_pred.append(y_pred)
        
        train_error = 1 - accuracy_score(y_train, y_train_pred)
        test_error = 1 - accuracy_score(y_test, y_test_pred)
        
        training_errors.append(train_error)
        test_errors.append(test_error)
        
        print(f'Lambda: {lambda_}, Training Error: {train_error}, Test Error: {test_error}')
    
    return training_errors, test_errors


import gzip
import pandas as pd

def read_digit_data(file_path):
    with gzip.open(file_path, 'rt') as f:
        df = pd.read_csv(f, delim_whitespace=True, header=None)
    return df


df_train = read_digit_data('train_data.gz')
df_test = read_digit_data('test_data.gz')

X_train = df_train.iloc[:, 1:].values
y_train = df_train.iloc[:, 0].values
X_test = df_test.iloc[:, 1:].values
y_test = df_test.iloc[:, 0].values

lambdas = [0.1, 1, 10, 100, 1000]
training_errors, test_errors = local_lda(X_train, y_train, X_test, y_test, lambdas)

print("\nFinal Training Errors:", training_errors)
print("Final Test Errors:", test_errors)



