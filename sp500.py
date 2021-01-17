import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

'''
    Implementation based on paper https://mpra.ub.uni-muenchen.de/62664/1/MPRA_paper_62664.pdf
    
    I added some extra variables, the OPH2 and OPL2 those seem to slightly improve the results
    In stead of ratios, this code works with absolute deltas, results were much better using absolute values
    
    Joost Bruneel
'''
def load_spx_data(filename):

    df = pd.read_csv(filename, sep=",")

    df["High1"]=df["High"].shift(1)
    df["Low1"]=df["Low"].shift(1)
    df["High2"]=df["High"].shift(2)
    df["Low2"]=df["Low"].shift(2)
    df["High3"]=df["High"].shift(3)
    df["Low3"]=df["Low"].shift(3)

    ## according to original paper relative change (ratio) of low en high
    '''df["OPH"] = (df["Open"]-df["High1"])/df["High1"]
    df["OPL"] = (df["Open"]-df["Low1"])/df["Low1"]
    df["OPH1"] = (df["Open"]-df["High2"])/df["High2"]
    df["OPL1"] = (df["Open"]-df["Low2"])/df["Low2"]
    df["OPH2"] = (df["Open"]-df["High3"])/df["High3"]
    df["OPL2"] = (df["Open"]-df["Low3"])/df["Low3"]

    df["GH"] = (df["High"]-df["High1"])/df["High1"]
    df["GL"] = (df["Low"]-df["Low1"])/df["Low1"]'''

    ## absolute differences give better results than relative
    df["OPH"] = (df["Open"] - df["High1"])
    df["OPL"] = (df["Open"] - df["Low1"])
    df["OPH1"] = (df["Open"] - df["High2"])
    df["OPL1"] = (df["Open"] - df["Low2"])
    df["OPH2"] = (df["Open"] - df["High3"])
    df["OPL2"] = (df["Open"] - df["Low3"])

    df["GH"] = (df["High"] - df["High1"])
    df["GL"] = (df["Low"] - df["Low1"])

    df["High_dir"] = df.apply(lambda row: 1 if row["High"]>=row["High1"] else -1, axis=1)
    df["Low_dir"] = df.apply(lambda row: 1 if row["Low"] >= row["Low1"] else -1, axis=1)


    df = df.dropna()

    return df


def regression(X,y, num_test_samples=100):

    X_train = X[0:-num_test_samples]
    y_train = y[0:-num_test_samples]

    X_test = X[-num_test_samples:]
    y_test = list(y[-num_test_samples:])

    lr = LinearRegression()
    lr.fit(X_train, list(y_train))

    y_pred = lr.predict(X_test)

   
    return y_pred,y_test

def classification(X, y, normalize=False, num_test_samples=100):

    Xs = StandardScaler().fit_transform(X) if normalize else X

    X_train = Xs[0:-num_test_samples]
    y_train = y[0:-num_test_samples]

    X_test = Xs[-num_test_samples:]
    y_test = list(y[-num_test_samples:])

    '''classifier = AdaBoostClassifier(n_estimators=60, random_state=0)
    classifier = SVC(kernel='rbf, C=0.1, gamma=0.1)
    classifier = MLPClassifier(alpha=0.7,max_iter=1000)
    classifier = GaussianProcessClassifier()
    classifier = QuadraticDiscriminantAnalysis()
    classifier = SGDClassifier(max_iter=1000,tol=1e-5)
    classifier = RidgeClassifier(alpha=0.5)
    classifier = GaussianProcessClassifier()'''

    classifier = SVC(kernel='poly', C=0.11, gamma=3, degree=1)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    return y_pred,y_test


def calc_result_stats(y_pred, y_test):
    count_success_lr = 0
    sum_y = 0
    sum_diff = 0
    for i in range(len(y_test)):
        sum_y = sum_y + abs(y_test[i])
        sum_diff = sum_diff + abs(y_test[i] - y_pred[i])
        if y_pred[i] * y_test[i] >= 0:
            count_success_lr = count_success_lr + 1

    return count_success_lr/len(y_test),sum_y/len(y_test),sum_diff/len(y_test)

def plot():

    fig = plt.figure()
    ax=fig.add_axes([0.1,0.1,0.80,0.8])
    ax.scatter(y_test_high, y_pred_high, color='r')
    ax.scatter(y_test_low, y_pred_low, color='b')

    plt.legend(["High", "Low"])
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    ax.set_title('SPX points day ahead up/down (Actual vs Predicted)')
    plt.plot(y_test_high,y_test_high)
    plt.plot(y_test_low,y_test_low)



    plt.show()

### Download from https://finance.yahoo.com/quote/%5EGSPC/history?period1=1262304000&period2=1610841600&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true
df = load_spx_data("C:\\Users\\joost\\Downloads\GSPC.csv")

df_h = df[df["Open"] <= df["High1"]]
df_l = df[df["Open"] >= df["Low1"]]

# filter trivial cases where open is outside the high/low area
X_h = df_h[["OPH","OPL","OPH1","OPL1","OPH2","OPL2"]]
X_l = df_l[["OPH","OPL","OPH1","OPL1","OPH2","OPL2"]]

y_gh = df_h["GH"]
y_gl = df_l["GL"]

num_test_samples = 300

y_pred_high, y_test_high = regression(X_h, y_gh, num_test_samples)
y_pred_low, y_test_low = regression(X_l, y_gl, num_test_samples)

y_pred_high_c, y_test_high_c = classification(X_h, df_h["High_dir"], True, num_test_samples)
y_pred_low_c, y_test_low_c = classification(X_l, df_l["Low_dir"], True, num_test_samples)

acc_high, avg_yh, mae = calc_result_stats(y_pred_high, y_test_high)
acc_low, avg_yl, mae = calc_result_stats(y_pred_low, y_test_low)
acc_class_high,_,_=calc_result_stats(y_pred_high_c, y_test_high_c)
acc_class_low,_,_=calc_result_stats(y_pred_low_c, y_test_low_c)

print("Total data samples",len(df),". Number of test samples",num_test_samples)
print("High direction with linear regression accuracy", acc_high)
print("Low direction with linear regression accuracy", acc_low)
print("High direction with SV classification accuracy", acc_class_high)
print("Low direction with SV classification accuracy", acc_class_low)

plot()
