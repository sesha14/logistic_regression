import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import math

#sigmoid function depiction
def sigmoid(z):
     return 1.0 / (1.0 + np.exp(-z))
z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)
plt.plot(z, phi_z)
plt.axvline(0.0, color='k')
plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
plt.axhline(y=0.5, ls='dotted', color='k')
plt.yticks([0.0, 0.5, 1.0])
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
plt.show()

#X,y values initialization
from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris.data[:, [0, 1]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#preprocessing
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


#visualize the data
print('data visualization')
plt.scatter(X[:50, 0], X[:50, 1],
 color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
 color='blue', marker='x', label='versicolor')
plt.scatter(X[100:150, 0], X[100:150, 1],
 color='red', marker='^', label='virginica')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc='upper left')
plt.show()

#split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#train and predict using logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)
y_pred=lr.predict(X_test_std)


#metrics valuation
print('LR Accuracy: %.2f' % accuracy_score(y_test, y_pred))


#decision boundary
print('decision region for Logistic regression')
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier,test_idx=None, resolution=0.02):
   # setup marker generator and color map
   markers = ('s', 'x', 'o', '^', 'v')
   colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
   cmap = ListedColormap(colors[:len(np.unique(y))])

   # plot the decision surface
   x1_min, x1_max = X[:,  0].min() - 1, X[:, 0].max() + 1
   x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
   xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
   np.arange(x2_min, x2_max, resolution))
   Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
   Z = Z.reshape(xx1.shape)
   plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
   plt.xlim(xx1.min(), xx1.max())
   plt.ylim(xx2.min(), xx2.max())

   # plot class samples
   for idx, cl in enumerate(np.unique(y)):
      plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
      alpha=0.8, c=cmap(idx),
      marker=markers[idx], label=cl,edgecolor='black')

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std,y=y_combined,classifier=lr,test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

#L1 Regularization
lr = LogisticRegression(penalty='l1', C=1.0)
lr.fit(X_train_std, y_train)
y_pred=lr.predict(X_test)
pred=lr.predict_proba(X_test)

#metrics
print('L1 Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('L1 logloss :',log_loss(y_test,pred))
   
#varying c values
weights, params = [], []
for c in np.arange(-4, 6):
   lr = LogisticRegression(C=10.**c, penalty='l1')
   lr.fit(X_train, y_train)
   weights.append(lr.coef_[0])
   params.append(10.**c)
weights = np.array(weights)
plt.plot(params, weights[:, 0],label='sepal length')
plt.plot(params, weights[:, 1], linestyle='--',label='sepal width')

#depict variation of c values
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
plt.show()


#L2 regularization
lr = LogisticRegression(penalty='l2', C=1.0)
lr.fit(X_train, y_train)
y_pred=lr.predict(X_test)
pred=lr.predict_proba(X_test)

#metrics
print('L2 Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('L2 logloss :',log_loss(y_test,pred))

#varying c values
weights, params = [], []
for c in np.arange(-4,6):
   lr = LogisticRegression(C=10.**c, random_state=1,penalty='l2')
   lr.fit(X_train, y_train)
   weights.append(lr.coef_[0])
   params.append(10.**c)
weights = np.array(weights)

#depict variation of c values
plt.plot(params, weights[:, 0],label='sepal length')
plt.plot(params, weights[:, 1], linestyle='--',label='sepal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()

#Cost vs epochs FUNCTIONS
def hypothesis(theta, X, n):
    h = np.ones((X.shape[0],1))
    theta = theta.reshape(1,n+1)
    for i in range(0,X.shape[0]):
        h[i] = 1 / (1 + math.exp(-float(np.matmul(theta, X[i]))))
    h = h.reshape(X.shape[0])
    return h

def BGD(theta, alpha, num_iters, h, X, y, n):
    theta_history = np.ones((num_iters,n+1))
    cost = np.ones(num_iters)
    for i in range(0,num_iters):
        theta[0] = theta[0] - (alpha/X.shape[0]) * sum(h - y)
        for j in range(1,n+1):
            theta[j]=theta[j]-(alpha/X.shape[0])*sum((h-y)
                               *X.transpose()[j])
        theta_history[i] = theta
        h = hypothesis(theta, X, n)
        cost[i]=(-1/X.shape[0])*sum(y*np.log(h)+(1-y)*np.log(1 - h))
    theta = theta.reshape(1,n+1)
    return theta, theta_history, cost

def logistic_regression(X, y, alpha, num_iters):
    n = X.shape[1]
    one_column = np.ones((X.shape[0],1))
    X = np.concatenate((one_column, X), axis = 1)
    # initializing the parameter vector...
    theta = np.zeros(n+1)
    # hypothesis calculation....
    h = hypothesis(theta, X, n)
    # returning the optimized parameters by Gradient Descent...
    theta,theta_history,cost = BGD(theta,alpha,num_iters,h,X,y,n)
    return theta, theta_history, cost

# calling the principal function with learning_rate = 0.001 and 
theta,theta_history,cost=logistic_regression(X_train,y_train,0.001, 100000)
cost = list(cost)
n_iterations = [x for x in range(1,100001)]
plt.plot(n_iterations, cost)
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.show()
