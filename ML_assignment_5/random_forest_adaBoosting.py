import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
import numpy as np
import matplotlib.pylab as plt 
from scipy import stats
import collections
#%matplotlib inline 
class ThreesandEights:
    """
    Class to store MNIST 3s and 8s data
    """

    def __init__(self, location):

        import pickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')

        # Split the data set 
        x_train, y_train, x_test, y_test = pickle.load(f)
                
        # Extract only 3's and 8's for training set 
        self.x_train = x_train[np.logical_or(y_train== 3, y_train == 8), :]
        self.y_train = y_train[np.logical_or(y_train== 3, y_train == 8)]
        self.y_train = np.array([1 if y == 8 else -1 for y in self.y_train])
        
        # Shuffle the training data 
        shuff = np.arange(self.x_train.shape[0])
        np.random.shuffle(shuff)
        self.x_train = self.x_train[shuff,:]
        self.y_train = self.y_train[shuff]

        # Extract only 3's and 8's for validation set 
        self.x_test = x_test[np.logical_or(y_test== 3, y_test == 8), :]
        self.y_test = y_test[np.logical_or(y_test== 3, y_test == 8)]
        self.y_test = np.array([1 if y == 8 else -1 for y in self.y_test])
        
        f.close()

def view_digit(ex, label=None, feature=None):
    """
    function to plot digit examples 
    """
    if label: print("true label: {:d}".format(label))
    img = ex.reshape(21,21)
    col = np.dstack((img, img, img))
    if feature is not None: col[feature[0]//21, feature[0]%21, :] = [1, 0, 0]
    plt.imshow(col)
    plt.xticks([]), plt.yticks([])
    #plt.show()
    
data = ThreesandEights("ML_assignment_5/data/mnist21x21_3789.pklz")

view_digit(data.x_train[0], data.y_train[0])

class AdaBoost:
    def __init__(self, n_learners=20, base=DecisionTreeClassifier(max_depth=3), random_state=1234):
        np.random.seed(42)
        self.n_learners = n_learners 
        self.base = base
        self.alpha = np.zeros(self.n_learners)
        self.learners = []
    
    def fit(self, X_train, y_train):
        w = np.ones(len(y_train), dtype=np.float128)
        w /= np.sum(w)
        self.learners = []
        self.alpha = np.zeros(self.n_learners)
        for k in range(self.n_learners):
            h = clone(self.base)
            h.fit(X_train, y_train, sample_weight=w)
            y_pred = h.predict(X_train)
            errk = self.error_rate(y_train, y_pred, w)
            # Avoid division by zero
            errk = np.clip(errk, 1e-10, 1-1e-10)
            alpha_k = 0.5 * np.log((1-errk)/errk)
            self.alpha[k] = alpha_k
            self.learners.append(h)
            # Update weights
            w *= np.exp(-alpha_k * y_train * y_pred)
            w /= np.sum(w)
        return self
    
    def error_rate(self, y_true, y_pred, weights):
        # Weighted error rate: sum weights where prediction is wrong, divided by total weights
        # AdaBoost expects labels to be -1 and 1, not 0 and 1
        # If labels are 0/1, convert to -1/1
        y_true_adj = np.where(y_true == 0, -1, y_true)
        y_pred_adj = np.where(y_pred == 0, -1, y_pred)
        return np.sum(weights * (y_true_adj != y_pred_adj)) / np.sum(weights)
    
    def predict(self, X):
        """
        Adaboost prediction for new data X.
        Returns: yhat (ndarray): [n_samples] ndarray of predicted labels {-1,1}
        """
        yhat = np.zeros(X.shape[0], dtype=np.float128)
        for k in range(self.n_learners):
            yhat += self.alpha[k] * self.learners[k].predict(X)
        yhat = np.sign(yhat)
        return yhat
    
    def score(self, X, y):
        yhat = self.predict(X)
        return np.mean(yhat == y)
    
    def staged_score(self, X, y):
        scores = []
        fx = np.zeros(X.shape[0], dtype=np.float128)
        for k in range(self.n_learners):
            fx += self.alpha[k] * self.learners[k].predict(X)
            yhat = np.sign(fx)
            scores.append(np.mean(yhat == y))
        return np.array(scores)
    

    # Sample test for Adaboost error rate function. 
import pytest

y_true = [-1, 1, 1, -1, 1, -1, -1]
y_pred = [-1, 1, 1, 1, 1, -1, 1]
w = np.ones(len(y_true))
w /= np.sum(w)

clf = AdaBoost() 
err_rate = clf.error_rate(y_true, y_pred, w)
assert pytest.approx(err_rate, 0.01) == 0.2857, "Check the error_rate function."

# Sample test for Adaboost fit function. 

sample_data = np.load('ML_assignment_5/data/train.npz') 
sample_X = sample_data['X']
sample_y = sample_data['y']
test_model = AdaBoost(n_learners=5).fit(sample_X,sample_y)
t_alpha = [1.94591015, 2.14179328, 2.48490665, 2.42209354, 3.1732565]
assert pytest.approx(test_model.alpha, 0.01) == t_alpha, "Check the fit function"

# use fit function to fit Adaboost classifier called clf with 150 base decision stumps
clf = AdaBoost(n_learners=150, base=DecisionTreeClassifier(max_depth=1))
clf.fit(data.x_train, data.y_train)

# print out predictions on the training set 
train_predict = clf.predict(data.x_train)
print(train_predict)
print(max(train_predict))

# Compute staged scores for train and test sets
train_scores = clf.staged_score(data.x_train, data.y_train)
test_scores = clf.staged_score(data.x_test, data.y_test)

# Compute misclassification error
train_error = 1 - train_scores
test_error = 1 - test_scores

# Plot misclassification error
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10,6))
# plt.plot(range(1, len(train_error)+1), train_error, label='Train Error', lw=2)
# plt.plot(range(1, len(test_error)+1), test_error, label='Test Error', lw=2)
# plt.xlabel('Number of Boosting Iterations', fontsize=14)
# plt.ylabel('Misclassification Error', fontsize=14)
# plt.title('AdaBoost Misclassification Error vs Number of Iterations', fontsize=16)
# plt.legend()
# plt.grid(alpha=0.3)
# plt.tight_layout()
#plt.show()


class RandomForest():
    
    def __init__(self, x, y, sample_sz, n_trees=200, n_features='sqrt', max_depth=10, min_samples_leaf=5):
        """
        Create a new random forest classifier.
        
        Args:
            x : Input Feature vector
            y : Corresponding Labels
            sample_sz : Sample size
            n_trees : Number of trees to ensemble
            n_features : Method to select subset of features 
            max_depth : Maximum depth of the trees in the ensemble
            min_sample_leaf : Minimum number of samples per leaf 
        """
        np.random.seed(12)
        if n_features == 'sqrt':
            self.n_features = int(np.sqrt(x.shape[1]))
        elif n_features == 'log2':
            self.n_features = int(np.log2(x.shape[1]))
        else:
            self.n_features = n_features
        print(self.n_features, "sha: ",x.shape[1])  
        self.features_set = []
        self.x, self.y, self.sample_sz, self.max_depth, self.min_samples_leaf  = x, y, sample_sz, max_depth, min_samples_leaf
        self.trees = [self.create_tree(i) for i in range(n_trees)]

    def create_tree(self,i):
        """
        create a single decision tree classifier
        """
        
        idxs = np.random.permutation(len(self.y))[:self.sample_sz]
        idxs = np.asarray(idxs)

        f_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        f_idxs = np.asarray(f_idxs)
        
        
        if i==0:
            self.features_set = np.array(f_idxs, ndmin=2)
        else:
            self.features_set = np.append(self.features_set, np.array(f_idxs,ndmin=2),axis=0)
        
        # TODO: build a decision tree classifier and train it with x and y that is a subset of data (use idxs and f_idxs)
        
        # your code here
     # Innitiate a DecisionTreeClassifier object
        clf = DecisionTreeClassifier(
            max_depth=self.max_depth, 
            min_samples_leaf=self.min_samples_leaf)
        
        # Ctreate a subset of the data using idxs and f_idxs
        x, y = self.x[idxs][:, f_idxs], self.y[idxs]
        
        # Train the classifier with the subset of data
        clf.fit(x, y)
        
        return clf        

       
    def predict(self, x):
        """
        Predict labels for input features x using the trained random forest.
        The prediction is based on a majority vote from all individual trees.
        """
        # A list to hold predictions from all trees
        predictions = []
        
        # Iterate through each tree and its corresponding feature set
        for i, tree in enumerate(self.trees):
            # Select the relevant features from the input data for this tree
            f_idxs = self.features_set[i]
            x_subset = x[:, f_idxs]
            
            # Get predictions from the current tree and add to the list
            tree_predictions = tree.predict(x_subset)
            predictions.append(tree_predictions)
        
        # Convert the list of predictions to a numpy array for easier manipulation
        predictions = np.array(predictions)
        
        # Transpose the array to have predictions for each sample in a column
        # e.g., predictions[0, :] are all tree predictions for the first sample
        final_predictions = np.zeros(x.shape[0])
        for j in range(x.shape[0]):
            # Get predictions for the j-th sample from all trees
            sample_predictions = predictions[:, j]
            # Find the most common prediction (majority vote)
            final_predictions[j] = collections.Counter(sample_predictions).most_common(1)[0][0]
            
        return final_predictions
    
    def score(self, X, y):
        
        # TODO: Compute the score using the predict function and true labels y
        
        # your code here
        # Get the model's predictions
        y_pred = self.predict(X)
        
        # Compare predictions with true labels and calculate accuracy
        accuracy = np.mean(y_pred == y)
        
        return accuracy        


# Example usage for MNIST 3 vs 8 classification:
#rf = RandomForest(data.x_train, data.y_train, sample_sz=1000, n_trees=100, n_features='sqrt', max_depth=10, min_samples_leaf=5)
sample_sz= int(len(data.x_train)*0.5)
rf = RandomForest(data.x_train, data.y_train, sample_sz, n_trees=100, n_features='sqrt', max_depth=5, min_samples_leaf=15)
train_score = rf.score(data.x_train, data.y_train)
pred_score = rf.score(data.x_test, data.y_test)
print('Train misclassification error:', 1 - train_score)
print('Misclassification error on test data : %0.3f'%pred_score)

# Check label distribution in train and test sets
print('Train label distribution:', np.unique(data.y_train, return_counts=True))
print('Test label distribution:', np.unique(data.y_test, return_counts=True))