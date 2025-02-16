########## -------- LIBRARIES -------- ##########

import pandas as pd
import numpy as np

########## -------- DATA PREPARATION AND DATA CLEANING -------- ##########

data = pd.read_csv('/Users/sarasartini/Desktop/MAGISTRALE DATA SCIENCE/Machine Learning/Progetto Machine Learning/MushroomDataset/secondary_data.csv', delimiter=';')
print(f"Original dataset dimensions: {data.shape}")

### Checking for missing values and duplicates BEFORE cleaning
missing_percent = data.isnull().mean() * 100
rows_with_missing = data.isnull().any(axis=1).sum()
duplicate_rows = data.duplicated().sum()

print("\nBEFORE CLEANING:")
print("Percentage of missing values per column:")
print(missing_percent[missing_percent > 0])  
print(f"\nNumber of rows with at least one missing value: {rows_with_missing}")
print(f"Number of duplicate rows: {duplicate_rows}")

# Storing initial dataset shape
initial_rows, initial_columns = data.shape

### Spliting the dataset into TRAINING (80%) and TESTING (20%)
np.random.seed(42)
mask = np.random.rand(len(data)) < 0.8
df_train = data[mask]
df_test = data[~mask]

### Removing columns with missing values > 40%
missing_percent_train = df_train.isnull().mean()
columns_to_drop = missing_percent_train[missing_percent_train > 0.4].index  
df_train_cleaned = df_train.drop(columns=columns_to_drop).dropna().drop_duplicates()
df_test_cleaned = df_test.drop(columns=columns_to_drop).dropna().drop_duplicates()

# Computing how many rows and columns were dropped
dropped_rows_train = df_train.shape[0] - df_train_cleaned.shape[0]
dropped_rows_test = df_test.shape[0] - df_test_cleaned.shape[0]
dropped_columns = len(columns_to_drop)

# Checking missing values and duplicates AFTER cleaning
missing_percent_after = df_train_cleaned.isnull().mean() * 100
rows_with_missing_after = df_train_cleaned.isnull().any(axis=1).sum()
duplicate_rows_after = df_train_cleaned.duplicated().sum()

print("\nAFTER CLEANING:")
print("Percentage of missing values per column:")
print(missing_percent_after[missing_percent_after > 0])  
print(f"Number of rows with at least one missing value: {rows_with_missing_after}")
print(f"Number of duplicate rows: {duplicate_rows_after}")

# Displaying summary of dropped data to see how many columns and rows are dropped
print("\nData Cleaning Summary:")
print(f"Columns dropped: {dropped_columns}")
print(f"Rows dropped from training set: {dropped_rows_train}")
print(f"Rows dropped from test set: {dropped_rows_test}")

# Displaying final dataset dimensions
print(f"\nFinal training dataset dimensions: {df_train_cleaned.shape}")
print(f"Final test dataset dimensions: {df_test_cleaned.shape}")

### Identifying categorical and numerical features
categorical_features = df_train_cleaned.select_dtypes(include=['object']).columns.tolist()
numerical_features = df_train_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Define feature types (needed for tree splitting)
feature_types = {col: 'numerical' if col in numerical_features else 'categorical' for col in df_train_cleaned.columns if col != 'class'}

### Separate features (X) from the target variable (y)
X_train = df_train_cleaned.drop("class", axis=1)
y_train = (df_train_cleaned["class"] == "p").astype(int).values
X_test = df_test_cleaned.drop("class", axis=1)
y_test = (df_test_cleaned["class"] == "p").astype(int).values


########## -------- BUILDING THE DECISION TREE -------- ##########

class Tree_Node:
    def __init__(self, is_leaf=False, test=None, left=None, right=None, prediction=None):
        self.is_leaf = is_leaf
        self.test = test
        self.left = left
        self.right = right
        self.prediction = prediction

# Impurity Functions (scaled entropy, custom impurity and gini)
def scaled_entropy(p):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -(p / 2) * np.log2(p) - ((1 - p) / 2) * np.log2(1 - p)

def custom_impurity(p):
    return np.sqrt(p * (1 - p))

def gini_impurity(y):
    p = np.mean(y)
    return 2 * p * (1 - p)

def select_impurity_function(impurity_type="gini"):
    if impurity_type == "gini":
        return gini_impurity
    elif impurity_type == "scaled_entropy":
        return lambda y: scaled_entropy(np.mean(y))
    elif impurity_type == "custom_impurity":
        return lambda y: custom_impurity(np.mean(y))
    else:
        raise ValueError("Unknown impurity function")

def split_dataset(X, y, feature, threshold):
    '''Function to split the dataset based on a given 
    feature and theshold, handling both numerical and 
    categolical variables'''
    if feature in categorical_features:
        left_mask = X[feature] == threshold
    else:
        left_mask = X[feature] <= threshold
    right_mask = ~left_mask
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

def best_split(X, y, impurity_func):
    '''Function to identify the best feature and threshold
    for splitting, by evaluating different potential splits
    using a given impurity function'''
    best_score = float('inf')
    best_test = None
    best_left_X, best_left_y, best_right_X, best_right_y = None, None, None, None

    for feature in X.columns:
        unique_values = X[feature].unique()
        thresholds = unique_values if feature in categorical_features else np.percentile(unique_values, [25, 50, 75])

        for threshold in thresholds:
            left_X, left_y, right_X, right_y = split_dataset(X, y, feature, threshold)

            if len(left_y) == 0 or len(right_y) == 0:
                continue

            impurity = (len(left_y) / len(y)) * impurity_func(left_y) + (len(right_y) / len(y)) * impurity_func(right_y)

            if impurity < best_score:
                best_score = impurity
                best_test = (feature, threshold)
                best_left_X, best_left_y, best_right_X, best_right_y = left_X, left_y, right_X, right_y

    return best_test, best_left_X, best_left_y, best_right_X, best_right_y


class Decision_Tree:
    def __init__(self, max_depth=None, min_samples_split=2, impurity_type="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.impurity_func = select_impurity_function(impurity_type)
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if len(np.unique(y)) == 1 or len(y) < self.min_samples_split or (self.max_depth and depth >= self.max_depth):
            return Tree_Node(is_leaf=True, prediction=np.bincount(y).argmax())

        test, left_X, left_y, right_X, right_y = best_split(X, y, self.impurity_func)
        if test is None:
            return Tree_Node(is_leaf=True, prediction=np.bincount(y).argmax())

        return Tree_Node(is_leaf=False, test=test,
                        left=self._build_tree(left_X, left_y, depth + 1),
                        right=self._build_tree(right_X, right_y, depth + 1))

    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for _, x in X.iterrows()])

    def _predict_single(self, x, node):
        if node.is_leaf:
            return node.prediction
        feature, threshold = node.test
        if feature in categorical_features:
            branch = node.left if x[feature] == threshold else node.right
        else:
            branch = node.left if x[feature] <= threshold else node.right
        return self._predict_single(x, branch)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

########## -------- K-FOLD SPLIT -------- ##########

def stratified_kfold_split(X, y, k=5):
    np.random.seed(42)
    unique_classes, class_counts = np.unique(y, return_counts=True)
    folds = [[] for _ in range(k)]

    for label in unique_classes:
        class_indices = np.where(y == label)[0]  
        np.random.shuffle(class_indices) 
        for i, index in enumerate(class_indices):
            folds[i % k].append(index)
    folds = [np.array(fold) for fold in folds]
    return folds

########## -------- CROSS VALIDATION -------- ##########

def cross_validation(X, y, k=5, max_depth_list=[5, 10, 15], min_samples_split_list=[2, 5, 10], impurity_list=["gini", "scaled_entropy", "custom_impurity"]):

    folds = stratified_kfold_split(X, y, k)
    results = []

    for max_depth in max_depth_list:
        for min_samples_split in min_samples_split_list:
            for impurity in impurity_list:
                fold_results = []

                for i in range(k):
                    test_idx = folds[i]
                    train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

                    X_train, y_train = X.iloc[train_idx], y[train_idx]
                    X_test, y_test = X.iloc[test_idx], y[test_idx]

                    tree = Decision_Tree(max_depth=max_depth, min_samples_split=min_samples_split, impurity_type=impurity)
                    tree.fit(X_train, y_train)

                    train_accuracy = tree.accuracy(X_train, y_train)
                    test_accuracy = tree.accuracy(X_test, y_test)

                    train_error = 1 - train_accuracy  
                    test_error = 1 - test_accuracy

                    fold_results.append((train_accuracy, test_accuracy, train_error, test_error))

                mean_train_acc = np.mean([r[0] for r in fold_results])
                mean_test_acc = np.mean([r[1] for r in fold_results])
                mean_train_error = np.mean([r[2] for r in fold_results])
                mean_test_error = np.mean([r[3] for r in fold_results])

                results.append({
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "impurity": impurity,
                    "mean_train_accuracy": mean_train_acc,
                    "mean_test_accuracy": mean_test_acc,
                    "mean_train_error": mean_train_error,
                    "mean_test_error": mean_test_error
                })

    return pd.DataFrame(results)


########## -------- RUNNING CROSS VALIDATION -------- ##########

cv_results = cross_validation(X_train, y_train)

# Display results sorted by best test accuracy
print("\nHyperparameter Tuning with Cross-Validation:")
print(cv_results.sort_values("mean_test_accuracy", ascending=False))



############# -------- RANDOM FOREST CLASS -------- #############

class Random_Forest:
    def __init__(self, num_trees=10, max_depth=10, min_samples_split=2, min_samples_leaf=1, num_features=None, max_features = None, impurity = 'Gini'):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.num_features = num_features
        self.max_features = max_features
        self.impurity = impurity
        self.trees = []

    def fit(self, X, y, feature_types):
        self.trees = []
        for _ in range(self.num_trees):
            tree = Decision_Tree(
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split,
                impurity_type='gini'
            )

            X_sample, y_sample = self.bootstrap_samples(X, y)

            num_features = int(np.sqrt(X.shape[1])) if self.num_features is None else self.num_features
            feature_indices = np.random.choice(X.columns, num_features, replace=False)
            X_sample_selected = X_sample[feature_indices]
            selected_feature_types = {feature: feature_types[feature] for feature in feature_indices}

            tree.fit(X_sample_selected, y_sample)

            self.trees.append((tree, feature_indices))

    def bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_sample = X.iloc[indices].reset_index(drop=True)
        y_sample = y[indices]
        return X_sample, y_sample

    def majority_vote(self, predictions):
        n_samples = predictions.shape[0]
        majority_votes = []
        for i in range(n_samples):
            unique_values, counts = np.unique(predictions[i], return_counts=True)
            majority_votes.append(unique_values[np.argmax(counts)])
        return np.array(majority_votes)

    def predict(self, X):
        tree_predictions = []
        for tree, feature_indices in self.trees:
            X_selected = X[feature_indices]
            preds = tree.predict(X_selected)
            tree_predictions.append(preds)

        tree_predictions = np.array(tree_predictions).T
        final_predictions = self.majority_vote(tree_predictions)
        return final_predictions

############# -------- CROSS VALIDATION FOR RANDOM FOREST -------- #############

def cross_validation_forest(X, y, feature_types, k=5, num_trees_list=[5], max_depth_list=[10, 15], 
                            min_samples_split_list=[2, 10], max_features_list=[None, 3],
                            impurity_list=['gini', 'scaled_entropy', 'custom_impurity']):
    
    folds = stratified_kfold_split(X, y, k)
    results = []
    
    for num_trees in num_trees_list:
        for max_depth in max_depth_list:
            for min_samples_split in min_samples_split_list:
                for max_features in max_features_list:
                    for impurity in impurity_list:
                        fold_results = []
                        
                        for i in range(k):
                            test_idx = folds[i]
                            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
                            
                            X_train, y_train = X.iloc[train_idx], y[train_idx]
                            X_test, y_test = X.iloc[test_idx], y[test_idx]
                            
                            forest = Random_Forest(num_trees=num_trees, max_depth=max_depth, 
                                                   min_samples_split=min_samples_split, max_features=max_features, 
                                                   impurity=impurity)
                            
                            forest.fit(X_train, y_train, feature_types)
                            
                            y_train_pred = forest.predict(X_train)
                            y_test_pred = forest.predict(X_test)
                            
                            train_accuracy = np.mean(y_train_pred == y_train)
                            test_accuracy = np.mean(y_test_pred == y_test)
                            train_error = 1 - train_accuracy
                            test_error = 1 - test_accuracy
                            
                            fold_results.append((train_accuracy, test_accuracy, train_error, test_error))
                        
                        mean_train_acc = np.mean([r[0] for r in fold_results])
                        mean_test_acc = np.mean([r[1] for r in fold_results])
                        mean_train_error = np.mean([r[2] for r in fold_results])
                        mean_test_error = np.mean([r[3] for r in fold_results])
                        
                        results.append({
                            "num_trees": num_trees,
                            "max_depth": max_depth,
                            "min_samples_split": min_samples_split,
                            "max_features": max_features,
                            "impurity": impurity,
                            "mean_train_accuracy": mean_train_acc,
                            "mean_test_accuracy": mean_test_acc,
                            "mean_train_error": mean_train_error,
                            "mean_test_error": mean_test_error
                        })
    
    return pd.DataFrame(results)


############# -------- RUNNING CROSS VALIDATION FOR RANDOM FOREST -------- #############

cv_forest_results = cross_validation_forest(X_train, y_train, feature_types)

# Display results sorted by best test accuracy
print("\nHyperparameter Tuning for Random Forest with Cross-Validation:")
print(cv_forest_results.sort_values("mean_test_accuracy", ascending=False))