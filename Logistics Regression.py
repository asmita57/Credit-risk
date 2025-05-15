# Loading libraries
            import numpy as np
            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt
            from sklearn.model_selection import train_test_split
            from sklearn import metrics


#Importing the dataset
            df = pd.read_csv("Downloads/CreditRisk.csv")
            df
            df.shape
            df.dtypes
            df.describe(include ='object')
            # Getting the uniQue Values
            for val in df:
                print(val, " ", df[val].unique().shape)

# dealing with UniQue, or same value columns.
            df.drop("Loan_ID", axis=1, inplace=True)
            df.shape
# Converting column type
            df["Credit_History"] = df["Credit_History"].astype("object")
            df.Credit_History.value_counts()

# Counting the freQunecy of variables to check the baisness
            for val in df:
                if df[val].dtypes == "object":
                    print(df[val].value_counts())

# Quasi_constant : Now check Quasi_constan features means varibles which has same values for majority observation'''
            quasi_constant_feat = []
            for feature in df.columns:
                dominant = (df[feature].value_counts() / np.float(len(df))).sort_values(ascending=False).values[0]
                if dominant > 0.80:
                    quasi_constant_feat.append(feature)

            print(quasi_constant_feat)
df.columns
df.isnull().sum()

df.dropna(thresh=df.shape[0] * 0.75, how='all', axis=1, inplace=True)
df.shape
# Dealing with NaN.
            df.isnull().any()
            # df.isnull()
            df.isnull().sum()
            for val in df:
                print(val, " ", (df[val].isnull().sum() / df.shape[0]) * 100)

# No of rows getting affctected by removing na's
            no_of_rows = df[df.isna().sum(axis=1) >= 1].shape[0]
# % of rows getting affcted by removing Na's from column.
            print((no_of_rows / df.shape[0]) * 100)


#  In case want to impute Na's : Imputation according to data type
            def imputenull(data):
                for col in data.columns:
                    if data[col].dtypes == 'int64' or data[col].dtypes == 'float64':
                        data[col].fillna((data[col].mean()), inplace=True)
                    else:
                        data[col].fillna(data[col].value_counts().index[0], inplace=True)


            imputenull(df)
df.skew()
df["Credit_History"].mean()              ### Credit history is object thn how na got replaced by mean
df.dtypes
df.dtypes
# Now checking on the dataset after imputing the columns
             df.isnull().sum()

df.columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'tar_var']
# Making a copy of cleaned data
            data = df.copy()
# Bar plot for tg variable
            sns.catplot(x="Loan_Status", kind="count", data=df)
df.skew()
# Training & Test split of data
            df.columns
            tar_var = df['tar_var']
            df.drop("tar_var", axis=1, inplace=True)
df.dtypes
# Converting in dummies
             df_num = df.select_dtypes(include=['int64', 'float64'])
            df = pd.get_dummies(df, drop_first=False)
            df.shape

#------------Converting categorical data into numeric
from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
df.select_dtypes(include=["object"]).columns
df[df.select_dtypes(include=["object"]).columns]=df[df.select_dtypes(include=["object"]).columns].apply(lbl.fit_transform)
df.dtypes

df.columns
plt.figure(figsize=(15,15))
sns.heatmap(df.corr(),annot = True,fmt = ".2f",cbar = True)
plt.xticks(rotation=90)
plt.yticks(rotation = 0)

from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

svm = LinearSVC()
rfe = RFE(svm,8)
rfe.fit(df, tar_var)
rfe.support_
df.columns

df.drop("Loan_Amount_Term", axis=1, inplace=True)
df.drop("Dependents", axis=1, inplace=True)

dt = DecisionTreeClassifier
rfe_dt = RFE(dt,5)
rfe_dt.fit(df, tar_var)

rfe.support_
# Converting categorical variable into factor.
#             lst = df_num.columns
#             for val in df:
#                 if(val not in lst):
#                     df[val] = df[val].astype("object")

#Spliting now.
            x_train,x_test,y_train,y_test = train_test_split(df, tar_var, random_state = 10,test_size = 0.3)

            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression()
            model.fit(x_train, y_train)

# create logistic regression object
            reg = model.LogisticRegression()

# train the model using the training sets
            reg.fit(x_train, y_train)

# making predictions on the testing set
            y_pred = model.predict(x_test)
###--------------------------------------------Checking Accuracy
accuracy1=accuracy_score(y_test,y_pred)
accuracy1


sensitivity1=(metrics.recall_score(y_test, y_pred))
sensitivity1

precision1= (metrics.precision_score(y_test, y_pred))
precision1

F1 = 2 * (precision1 * sensitivity1) / (precision1 + sensitivity1)
F1

generate_model _report(y_test, y_pred)


print(accuracy1)
print(precision1)
print(sensitivity1)
print(F1)

# comparing actual response values (y_test) with predicted response values (y_pred)
            print("Logistic Regression model accuracy in %"):
            metrics.accuracy_score(y_test, y_pred) * 100
            print(metrics.confusion_matrix(y_test, y_pred))

# save confusion matrix and slice into four pieces---- deep diving into confusion matrix
            confusion = metrics.confusion_matrix(y_test, y_pred)
            print(confusion)
            #[row, column]
            TP = confusion[1, 1]
            TN = confusion[0, 0]
            FP = confusion[0, 1]
            FN = confusion[1, 0]
# Converting the categorical output into numerical output
            lst_test =[]
            for val in y_test:
                x = 1 if(val == "Y") else 0
                lst_test.append(x)

            lst_pred =[]
            for val in y_pred:
                x = 1 if(val == "Y") else 0
                lst_pred.append(x)



            print((TP + TN) / float(TP + TN + FP + FN))                 #Accuracy by calculation
            print(metrics.accuracy_score(lst_test, lst_pred))         # Confusion maytrix

            classification_error = (FP + FN) / float(TP + TN + FP + FN) #Error
            print(classification_error*100)
            print(1 - metrics.accuracy_score(lst_test, lst_pred))

            sensitivity = TP / float(FN + TP)
            print(sensitivity)
            print(metrics.recall_score(lst_test, lst_pred))

            specificity = TN / (TN + FP)
            print(specificity)

            false_positive_rate = FP / float(TN + FP)
            print(false_positive_rate)
            print(1 - specificity)

            precision = TP / float(TP + FP)
            print(precision)
            print(metrics.precision_score(lst_test, lst_pred))


metrics.accuracy_score(lst_test, lst_pred)
metrics.recall_score(lst_test, lst_pred)
metrics.precision_score(lst_test, lst_pred)
metrics.roc_auc_score(lst_test, lst_pred)
            """Receiver Operating Characteristic (ROC)"""
            # IMPORTANT: first argument is true values, second argument is predicted values
            # roc_curve returns 3 objects fpr, tpr, thresholds
            # fpr: false positive rate
            # tpr: true positive rate
            fpr, tpr, thresholds = metrics.roc_curve(lst_test, lst_pred)
            plt.plot(fpr, tpr)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.rcParams['font.size'] = 12
            plt.title('ROC curve for Credit risk classifier')
            plt.xlabel('False Positive Rate (1 - Specificity)')
            plt.ylabel('True Positive Rate (Sensitivity)')
            plt.grid(True)


            """AUC - Area under Curve"""

            # AUC is the percentage of the ROC plot that is underneath the curve:
            # IMPORTANT: first argument is true values, second argument is predicted probabilities
            print(metrics.roc_auc_score(lst_test, lst_pred))


            # F1 Score FORMULA
            F1 = 2 * (precision * sensitivity) / (precision + sensitivity)
df.skew()

# calculate cross-validated AUC only for logistics
        from sklearn.model_selection import cross_val_score
        cross_val_score(reg, x_train, y_train, cv=10, scoring='roc_auc').mean()