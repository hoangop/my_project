import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score

def get_spam_dataset(filepath="tree_decisiom/data/spamdata.csv", test_split=0.1):
    '''
    Loads csv file, shuffles, and splits into train/test sets.
    Returns X_train, X_test, y_train, y_test, feature_names.
    '''
    # Read the CSV file
    df = pd.read_csv(filepath, delim_whitespace=True)
    feature_names = list(df.columns)
    
    # Assume the label column is named 'isSpam'
    #print(df.columns)
    X = df.drop('isSPAM', axis=1).values
    y = df['isSPAM'].values

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42, shuffle=True
    )
    
    return X_train, X_test, y_train, y_test, feature_names

def build_dt(data_X, data_y, max_depth = None, max_leaf_nodes =None):
    '''
    Hàm này thực hiện các bước sau:
    1. Xây dựng bộ phân loại cây quyết định bằng sklearn
    2. Huấn luyện nó với dữ liệu được cung cấp.
    
    
    Arguments
        data_X - một np.ndarray
        data_y - np.ndarray
        max_depth - None nếu không giới hạn, nếu không thì là một số nguyên cho độ sâu tối đa
                mà cây có thể đạt tới.
        max_leaf_nodes - None nếu không giới hạn, nếu không thì là một số nguyên
                cho số nút lá tối đa.
    
    Returns:
        Một đối tượng DecisionTreeClassifier đã được huấn luyện
    '''
    

    clf = DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random_state=42) 

    clf.fit(data_X, data_y)
    
    return clf


def calculate_precision(y_true, y_pred, pos_label_value=1.0):
    '''
    Hàm này chấp nhận các nhãn và các dự đoán, sau đó
    tính toán precision cho một bộ phân loại nhị phân.
    
    Args
        y_true: np.ndarray
        y_pred: np.ndarray
        
        pos_label_value: (float) số đại diện cho nhãn tích cực
        trong các mảng y_true và y_pred. Các số khác sẽ được coi là
        lớp không tích cực cho bộ phân loại nhị phân.
    
    Returns precision dưới dạng một số thập phân giữa 0.0 và 1.0
    '''
    # Tìm các dự đoán dương tính (positive predictions)
    true_positives = np.sum((y_true == pos_label_value) & (y_pred == pos_label_value))
    false_positives = np.sum((y_true != pos_label_value) & (y_pred == pos_label_value))
    
    # Tổng số dự đoán dương tính
    total_positives = true_positives + false_positives
    
    # Ngăn chặn lỗi chia cho 0
    if total_positives == 0:
        return 0.0
    
    # Tính toán precision
    precision = true_positives / total_positives
    
    return precision

def calculate_recall(y_true, y_pred, pos_label_value=1.0):
    '''
    Hàm này chấp nhận các nhãn và các dự đoán, sau đó
    tính toán recall cho một bộ phân loại nhị phân.
    
    Args
        y_true: np.ndarray
        y_pred: np.ndarray
        
        pos_label_value: (float) số đại diện cho nhãn tích cực
        trong các mảng y_true và y_pred. Các số khác sẽ được coi là
        lớp không tích cực cho bộ phân loại nhị phân.
    
    Returns recall dưới dạng một số thập phân giữa 0.0 và 1.0
    '''
    # Tìm các dự đoán dương tính thực tế (actual positives)
    true_positives = np.sum((y_true == pos_label_value) & (y_pred == pos_label_value))
    false_negatives = np.sum((y_true == pos_label_value) & (y_pred != pos_label_value))
    
    # Tổng số trường hợp dương tính thực tế
    total_actual_positives = true_positives + false_negatives
    
    # Ngăn chặn lỗi chia cho 0
    if total_actual_positives == 0:
        return 0.0
    
    # Tính toán recall
    recall = true_positives / total_actual_positives
    
    return recall


test_split = 0.1 # mặc định test_split; có thể thay đổi nếu muốn; đảm bảo rằng biến này được sử dụng như một đối số cho hàm của bạn
# Gọi hàm và gán kết quả cho các biến
X_train, X_test, y_train, y_test, label_names = get_spam_dataset(filepath="tree_decisiom/data/spamdata.csv", test_split=test_split)
# In kích thước của các tập dữ liệu để xác minh
# print("Kích thước X_train:", X_train.shape)
# print("Kích thước X_test:", X_test.shape)
# print("Kích thước y_train:", y_train.shape)
# print("Kích thước y_test:", y_test.shape)
# print("Tên các đặc trưng:", label_names)

dt_model = build_dt(X_train, y_train)

# Dự đoán trên tập dữ liệu kiểm tra
y_pred = dt_model.predict(X_test)

# Tính toán các chỉ số hiệu suất
# Precision: độ chính xác của các dự đoán tích cực
# Recall: khả năng tìm thấy tất cả các trường hợp tích cực
# pos_label=1 vì isSpam = 1 (SPAM) là nhãn tích cực
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)

# In kết quả
print("\nResults for the decision tree with no limit on depth or leaf nodes:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Max depth: {dt_model.get_depth()}")

# # Tính toán các chỉ số hiệu suất bằng hàm tự tạo
# precision_custom = calculate_precision(y_test, y_pred, pos_label_value=1.0)
# recall_custom = calculate_recall(y_test, y_pred, pos_label_value=1.0)

# # Sample Test cell 
# ut_true = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
# ut_pred = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0])
# prec = calculate_precision(ut_true, ut_pred, 1.0)
# recall = calculate_recall(ut_true, ut_pred, 1.0)
# assert prec == 0.6, "Check the precision value returned from your calculate_precision function."
# assert recall == 0.75, "Check the recall value returned from your calculate_recall function."

# TODO : Hoàn thành nhiệm vụ con đầu tiên cho max_depth

# Build a Decision Tree with max_depth=2
dt_model_depth_2 = build_dt(X_train, y_train, max_depth=2)

#Predict on the test set
y_pred_depth_2 = dt_model_depth_2.predict(X_test)

# Calculate precision and recall for the model with max_depth=2 using the custom functions
precision_depth_2 = calculate_precision(y_test, y_pred_depth_2, pos_label_value=1.0)
recall_depth_2 = calculate_recall(y_test, y_pred_depth_2, pos_label_value=1.0)

#Report the results
print("\nResults for the decision tree with max depth =2:")
print(f"Precision: {precision_depth_2:.4f}")
print(f"Recall: {recall_depth_2:.4f}")
print(f"Max depth: {dt_model_depth_2.get_depth()}")

# TODO : Complete the second subtask for max_leaf_nodes

# Modifying max_leaf_nodes
# Building a model with a shallow max_leaf_nodes of 4.
dt_model_leaf_4 = build_dt(X_train, y_train, max_leaf_nodes=4)

# Predicting on the test set.
y_pred_leaf_4 = dt_model_leaf_4.predict(X_test)

# Calculating performance metrics.
precision_leaf_4 = calculate_precision(y_test, y_pred_leaf_4, pos_label_value=1.0)
recall_leaf_4 = calculate_recall(y_test, y_pred_leaf_4, pos_label_value=1.0)

# Reporting the metrics and the tree's depth.
print("\nResults for the decision tree with max_leaf_nodes = 4:")
print(f"Precision: {precision_leaf_4:.4f}")
print(f"Recall: {recall_leaf_4:.4f}")
print(f"Depth of the tree: {dt_model_leaf_4.get_depth()}")



# Part D - Cost Complexity Pruning
dt = build_dt(X_train, y_train)

# Calculate the pruning path
path = dt.cost_complexity_pruning_path(X_train,y_train) #post pruning
ccp_alphas, impurities = path.ccp_alphas, path.impurities

clfs = [] # VECTOR CONTAINING CLASSIFIERS FOR DIFFERENT ALPHAS
# TODO: iterate over ccp_alpha values
# Iterate over each alpha and create a pruned tree
for ccp_alpha in ccp_alphas:
    # Build a new tree with the current alpha
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    # Fit the tree to the training data
    clf.fit(X_train, y_train)
    # Append the trained classifier to the list
    clfs.append(clf)
    
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
      clfs[-1].tree_.node_count, ccp_alphas[-1]))

# TODO: next, generate the train and test scores and plot the variation in these scores with increase in ccp_alpha
# The code for plotting has been provided; edit the train_scores and test_scores variables for the right plot to be generated
train_scores = []
test_scores = []

# your code here
# Iterate over each classifier in the list
for clf in clfs:
    # Calculate and append the training accuracy
    train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
    # Calculate and append the testing accuracy
    test_scores.append(accuracy_score(y_test, clf.predict(X_test)))


fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show() 
