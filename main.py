import pandas as pd
import numpy as np
import os
from tabulate import tabulate
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

#%% get data

print(os.listdir("./data/"))
application_train = pd.read_csv('./data/application_train.csv')
application_test = pd.read_csv('./data/application_test.csv')
POS_CASH_balance = pd.read_csv('./data/POS_CASH_balance.csv')
bureau_balance = pd.read_csv('./data/bureau_balance.csv')
previous_application = pd.read_csv('./data/previous_application.csv')
installments_payments = pd.read_csv('./data/installments_payments.csv')
credit_card_balance = pd.read_csv('./data/credit_card_balance.csv')
bureau = pd.read_csv('./data/bureau.csv')


#%% check data

df_list_names = ['application_train','POS_CASH_balance',
                'bureau_balance','previous_application',
                'installments_payments','credit_card_balance',
                'bureau'
                 ]


def f_shape_view(df_list):
    v_temp_list = []
    for i in df_list_names:
            v_temp_list.append([i,globals()[i].shape[0],globals()[i].shape[1]])
    return tabulate(v_temp_list,headers={'dataframe','rows','columns'})

print(f_shape_view(df_list_names))

# for i in df_list_names:
#     print('{s1} - rows: {s2} - columns: {s3}'.format(s1=i,s2=globals()[i].shape[0],s3=globals()[i].shape[1]))

#%% check missing data
def f_missing_data_view(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return tabulate(pd.concat([total, percent], axis=1, keys=['Total', 'Percent']),headers={'fields','total','percent'},tablefmt="psql")

for i in df_list_names:
    print('-------',i,'----------')
    print(f_missing_data_view(globals()[i]))

#%%check the unbalance

temp = application_train["TARGET"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
plt.figure(figsize = (6,6))
plt.title('Application loans repayed - train dataset')
sns.set_color_codes("pastel")
sns.barplot(x = 'labels', y="values", data=df)
locs, labels = plt.xticks()
plt.show()

#%% Explore the data

#Application data

def plot_stats(feature, label_rotation=False, horizontal_layout=True):
    temp = application_train[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index, 'Number of contracts': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = application_train[[feature, 'TARGET']].groupby([feature], as_index=False).mean()
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)

    if (horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 14))
    sns.set_color_codes("pastel")
    s = sns.barplot(ax=ax1, x=feature, y="Number of contracts", data=df1)
    if (label_rotation):
        s.set_xticklabels(s.get_xticklabels(), rotation=90)

    s = sns.barplot(ax=ax2, x=feature, y='TARGET', order=cat_perc[feature], data=cat_perc)
    if (label_rotation):
        s.set_xticklabels(s.get_xticklabels(), rotation=90)
    plt.ylabel('Percent of target with value 1 [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.show()


def plot_distribution(var):
    i = 0
    t1 = application_train.loc[application_train['TARGET'] != 0]
    t0 = application_train.loc[application_train['TARGET'] == 0]

    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    for feature in var:
        i += 1
        plt.subplot(2, 2, i)
        sns.kdeplot(t1[feature], bw=0.5, label="TARGET = 1")
        sns.kdeplot(t0[feature], bw=0.5, label="TARGET = 0")
        plt.ylabel('Density plot', fontsize=12)
        plt.xlabel(feature, fontsize=12)
        locs, labels = plt.xticks()
        plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()

plot_stats('NAME_CONTRACT_TYPE')
plot_stats('CODE_GENDER')
plot_stats('FLAG_OWN_CAR')
plot_stats('FLAG_OWN_REALTY')
plot_stats('NAME_FAMILY_STATUS',True, True)
plot_stats('CNT_CHILDREN')
plot_stats('CNT_FAM_MEMBERS',True)
plot_stats('NAME_INCOME_TYPE',False,False)
plot_stats('OCCUPATION_TYPE',True, False)
plot_stats('ORGANIZATION_TYPE',True, False)
plot_stats('NAME_EDUCATION_TYPE',True)
plot_stats('NAME_HOUSING_TYPE',True)
plot_stats('REG_REGION_NOT_LIVE_REGION')
plot_stats('REG_REGION_NOT_WORK_REGION')
plot_stats('REG_CITY_NOT_LIVE_CITY')
plot_stats('REG_CITY_NOT_WORK_CITY')

#%% distributions
def plot_distribution(feature,color):
    plt.figure(figsize=(10,6))
    plt.title("Distribution of %s" % feature)
    sns.distplot(application_train[feature].dropna(),color=color, kde=True,bins=100)
    plt.show()


# Plot distribution of multiple features, with TARGET = 1/0 on the same graph
def plot_distribution_comp(var, nrow=2):
    i = 0
    t1 = application_train.loc[application_train['TARGET'] != 0]
    t0 = application_train.loc[application_train['TARGET'] == 0]

    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(nrow, 2, figsize=(12, 6 * nrow))

    for feature in var:
        i += 1
        plt.subplot(nrow, 2, i)
        sns.kdeplot(t1[feature], bw=0.5, label="TARGET = 1")
        sns.kdeplot(t0[feature], bw=0.5, label="TARGET = 0")
        plt.ylabel('Density plot', fontsize=12)
        plt.xlabel(feature, fontsize=12)
        locs, labels = plt.xticks()
        plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()


plot_distribution('AMT_INCOME_TOTAL','green')
plot_distribution('AMT_CREDIT','blue')
plot_distribution('AMT_ANNUITY','tomato')
plot_distribution('AMT_GOODS_PRICE','brown')
plot_distribution('DAYS_BIRTH','blue')
plot_distribution('DAYS_EMPLOYED','red')
plot_distribution('DAYS_REGISTRATION','green')
plot_distribution('DAYS_ID_PUBLISH','blue')
#var = ['AMT_ANNUITY','AMT_GOODS_PRICE','DAYS_EMPLOYED', 'DAYS_REGISTRATION','DAYS_BIRTH','DAYS_ID_PUBLISH']
#plot_distribution_comp(var,nrow=3)

#%% Bureau Data
application_bureau_train = application_train.merge(bureau, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='inner')


def plot_b_stats(feature, label_rotation=False, horizontal_layout=True):
    temp = application_bureau_train[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index, 'Number of contracts': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = application_bureau_train[[feature, 'TARGET']].groupby([feature], as_index=False).mean()
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)

    if (horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 14))
    sns.set_color_codes("pastel")
    s = sns.barplot(ax=ax1, x=feature, y="Number of contracts", data=df1)
    if (label_rotation):
        s.set_xticklabels(s.get_xticklabels(), rotation=90)

    s = sns.barplot(ax=ax2, x=feature, y='TARGET', order=cat_perc[feature], data=cat_perc)
    if (label_rotation):
        s.set_xticklabels(s.get_xticklabels(), rotation=90)
    plt.ylabel('Percent of target with value 1 [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.show()

plot_b_stats('CREDIT_ACTIVE')
plot_b_stats('CREDIT_CURRENCY')
plot_b_stats('CREDIT_TYPE', True, True)

#%% CreditBureau distribution

def plot_b_distribution(feature, color):
    plt.figure(figsize=(10, 6))
    plt.title("Distribution of %s" % feature)
    sns.distplot(application_bureau_train[feature].dropna(), color=color, kde=True, bins=100)
    plt.show()

def is_outlier(points, thresh=3.5):

    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def plot_b_o_distribution(feature,color):
    plt.figure(figsize=(10,6))
    plt.title("Distribution of %s" % feature)
    x = application_bureau_train[feature].dropna()
    filtered = x[~is_outlier(x)]
    sns.distplot(filtered,color=color, kde=True,bins=100)
    plt.show()


# Plot distribution of multiple features, with TARGET = 1/0 on the same graph
def plot_b_distribution_comp(var, nrow=2):
    i = 0
    t1 = application_bureau_train.loc[application_bureau_train['TARGET'] != 0]
    t0 = application_bureau_train.loc[application_bureau_train['TARGET'] == 0]

    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(nrow, 2, figsize=(12, 6 * nrow))

    for feature in var:
        i += 1
        plt.subplot(nrow, 2, i)
        sns.kdeplot(t1[feature], bw=0.5, label="TARGET = 1")
        sns.kdeplot(t0[feature], bw=0.5, label="TARGET = 0")
        plt.ylabel('Density plot', fontsize=12)
        plt.xlabel(feature, fontsize=12)
        locs, labels = plt.xticks()
        plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()

# var = ['DAYS_CREDIT','CREDIT_DAY_OVERDUE','AMT_CREDIT_SUM','AMT_CREDIT_SUM_LIMIT']
# plot_b_distribution_comp(var, nrow=2)

plot_b_distribution('DAYS_CREDIT', 'green')
plot_b_distribution('CREDIT_DAY_OVERDUE','red')
plot_b_distribution('AMT_CREDIT_SUM','blue')
plot_b_o_distribution('AMT_CREDIT_SUM','blue')
plot_b_distribution('AMT_CREDIT_SUM_LIMIT','blue')

#%% Previous application data

application_prev_train = application_train.merge(previous_application, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='inner')


def plot_p_stats(feature, label_rotation=False, horizontal_layout=True):
    temp = application_prev_train[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index, 'Number of contracts': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = application_prev_train[[feature, 'TARGET']].groupby([feature], as_index=False).mean()
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)

    if (horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 14))
    sns.set_color_codes("pastel")
    s = sns.barplot(ax=ax1, x=feature, y="Number of contracts", data=df1)
    if (label_rotation):
        s.set_xticklabels(s.get_xticklabels(), rotation=90)

    s = sns.barplot(ax=ax2, x=feature, y='TARGET', order=cat_perc[feature], data=cat_perc)
    if (label_rotation):
        s.set_xticklabels(s.get_xticklabels(), rotation=90)
    plt.ylabel('Percent of target with value 1 [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.show()

plot_p_stats('NAME_CONTRACT_TYPE_y')
plot_p_stats('NAME_CASH_LOAN_PURPOSE', True, True)
plot_p_stats('NAME_CONTRACT_STATUS', True, True)
plot_p_stats('NAME_PAYMENT_TYPE', True, True)
plot_p_stats('NAME_CLIENT_TYPE')

#%% Feature Engine


# print(application_train.columns.values)
print(application_train.info(max_cols=122))


plot_stats('NAME_EDUCATION_TYPE')
plot_stats('NAME_FAMILY_STATUS')
plot_stats('REGION_POPULATION_RELATIVE')
plot_stats('DAYS_ID_PUBLISH')
plot_stats('OCCUPATION_TYPE')
plot_stats('REGION_RATING_CLIENT')
plot_stats('ORGANIZATION_TYPE')
plot_stats('FLOORSMAX_MEDI')
plot_stats('DAYS_LAST_PHONE_CHANGE')


# application_train = pd.get_dummies(application_train)
# application_test = pd.get_dummies(application_test)
#
# print("Training Features shape: ", application_train.shape)
# print("Testing Features shape: ", application_test.shape)
#
# train_labels = application_train['TARGET']
# application_train, application_test = application_train.align(application_test, join = 'inner', axis = 1)
#
# print("Training Features shape: ", application_train.shape)
# print("Testing Features shape: ", application_test.shape)
#
# application_train["TARGET"] = train_labels
# ​
# cust_data.head(15)

#%% Correlation

correlations = application_train.corr()['TARGET'].sort_values()

pd.set_option('display.max_columns', 500)

print(correlations)

# sns.heatmap(application_train, fmt = "0.2f", linewidths=0.5,annot_kws={"size":50},cmap='Reds')
# plt.tick_params(labelsize=50)
# plt.xticks(rotation=90)
# plt.yticks(rotation=0)
# plt.title('Correlation Heatmap');
# plt.show()

#%% WOE FUNCTION
pd.set_option('display.max_columns', 500)
cust_data = application_train[['NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','REGION_POPULATION_RELATIVE',
                                'DAYS_ID_PUBLISH','OCCUPATION_TYPE','REGION_RATING_CLIENT','ORGANIZATION_TYPE',
                                'FLOORSMAX_MEDI','DAYS_LAST_PHONE_CHANGE','TARGET'
                               ]]
# print(cust_data.head(15))

cust_data_null=pd.DataFrame(cust_data.isnull().sum()).reset_index()
cust_data_null.columns = ['column','null_cnt']

cust_data_unique=pd.DataFrame(cust_data.nunique()).reset_index()
cust_data_unique.columns = ['column','unique_cnt']

# print(cust_data_null)
# print(cust_data_unique)

cust_data_null_unigue = cust_data_null.merge(cust_data_unique, left_on='column', right_on='column', how='inner')

print(cust_data_null_unigue)

def calc_woe_iv(col):
    df = pd.DataFrame(columns=['values', 'total', 'good', 'bad', 'event_rate', 'non_event_rate', 'per_total_events',
                               'per_total_non_events', 'WOE', 'IV'])
    df['values'] = cust_data[col].unique()
    df.set_index('values', inplace=True)

    values = cust_data[col].unique()
    total_dict = dict(cust_data.groupby(col).size())
    col_target_dict = dict(cust_data.groupby([col, 'TARGET']).size())
    target_count = dict(cust_data.groupby(['TARGET']).size())

    for value in values:
        df.loc[value]['total'] = total_dict[value]
        if (value, 1) in col_target_dict:
            df.loc[value]['good'] = col_target_dict[(value, 1)]
        else:
            df.loc[value]['good'] = 0

        if (value, 0) in col_target_dict:
            df.loc[value]['bad'] = col_target_dict[(value, 0)]
        else:
            df.loc[value]['bad'] = 0

        if df.loc[value]['bad'] == 0:
            df = df.drop([value])

    df['event_rate'] = df['good'] / df['total']
    df['non_event_rate'] = df['bad'] / df['total']

    df['per_total_events'] = df['good'] / target_count[1]
    df['per_total_non_events'] = df['bad'] / target_count[0]

    df['WOE'] = np.log(df.per_total_events.astype('float64') / df.per_total_non_events.astype('float64'))
    df['IV'] = (df['per_total_events'] - df['per_total_non_events']) * df['WOE']

    return df

#%% check woe

# NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','REGION_POPULATION_RELATIVE',
#                                 'DAYS_ID_PUBLISH','OCCUPATION_TYPE','REGION_RATING_CLIENT','ORGANIZATION_TYPE',
#                                 'FLOORSMAX_MEDI','DAYS_LAST_PHONE_CHANGE','TARGET'



# iv_values = pd.DataFrame(columns = ['col_name','iv_value'])
# iv_values['col_name'] = cust_data.columns
# iv_values.set_index(['col_name'],inplace = True)
# iv_values.drop(['TARGET'],inplace = True)
# ORGANIZATION_TYPE_df = calc_woe_iv('ORGANIZATION_TYPE')
# iv_values.loc['ORGANIZATION_TYPE'] = ORGANIZATION_TYPE_df.IV.sum()
# print(iv_values.loc['ORGANIZATION_TYPE'])
# print(ORGANIZATION_TYPE_df)

# print(cust_data.describe())

#%% test

# print(cust_data.columns)

# print (cust_data['TARGET'])
# y = cust_data[['TARGET']]
# cust_data_x=cust_data.drop('TARGET')
# print(y)
# print(cust_data_x.columns)

#%% feature engineering
from sklearn.model_selection import train_test_split

# y = cust_data.pop('TARGET').values

y = cust_data[['TARGET']]
cust_data_x=cust_data.drop(['TARGET'],axis=1)
X_train, X_temp, y_train, y_temp = train_test_split(cust_data_x, y, stratify = y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, stratify = y_temp, test_size=0.5, random_state=42)
print('Shape of X_train:',X_train.shape)
print('Shape of X_val:',X_val.shape)
print('Shape of X_test:',X_test.shape)



# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

#%% reshape and clean data
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Seperation of columns into numeric and categorical columns
types = np.array([dt for dt in X_train.dtypes])
all_columns = X_train.columns.values
is_num = types != 'object'
num_cols = all_columns[is_num]
cat_cols = all_columns[~is_num]
# Featurization of numeric data
imputer_num = SimpleImputer(strategy='median')
X_train_num = imputer_num.fit_transform(X_train[num_cols])
X_val_num = imputer_num.transform(X_val[num_cols])
X_test_num = imputer_num.transform(X_test[num_cols])
scaler_num = StandardScaler()
X_train_num1 = scaler_num.fit_transform(X_train_num)
X_val_num1 = scaler_num.transform(X_val_num)
X_test_num1 = scaler_num.transform(X_test_num)
X_train_num_final = pd.DataFrame(X_train_num1, columns=num_cols)
X_val_num_final = pd.DataFrame(X_val_num1, columns=num_cols)
X_test_num_final = pd.DataFrame(X_test_num1, columns=num_cols)
# Featurization of categorical data
imputer_cat = SimpleImputer(strategy='constant', fill_value='MISSING')
X_train_cat = imputer_cat.fit_transform(X_train[cat_cols])
X_val_cat = imputer_cat.transform(X_val[cat_cols])
X_test_cat = imputer_cat.transform(X_test[cat_cols])
X_train_cat1= pd.DataFrame(X_train_cat, columns=cat_cols)
X_val_cat1= pd.DataFrame(X_val_cat, columns=cat_cols)
X_test_cat1= pd.DataFrame(X_test_cat, columns=cat_cols)
ohe = OneHotEncoder(sparse=False,handle_unknown='ignore')
X_train_cat2 = ohe.fit_transform(X_train_cat1)
X_val_cat2 = ohe.transform(X_val_cat1)
X_test_cat2 = ohe.transform(X_test_cat1)
cat_cols_ohe = list(ohe.get_feature_names(input_features=cat_cols))
X_train_cat_final = pd.DataFrame(X_train_cat2, columns = cat_cols_ohe)
X_val_cat_final = pd.DataFrame(X_val_cat2, columns = cat_cols_ohe)
X_test_cat_final = pd.DataFrame(X_test_cat2, columns = cat_cols_ohe)
# Final complete data
X_train_final = pd.concat([X_train_num_final,X_train_cat_final], axis = 1)
X_val_final = pd.concat([X_val_num_final,X_val_cat_final], axis = 1)
X_test_final = pd.concat([X_test_num_final,X_test_cat_final], axis = 1)
print(X_train_final.shape)
print(X_val_final.shape)
print(X_test_final.shape)

#%% PLOT function

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(test_y, predicted_y):
    # Confusion matrix
    C = confusion_matrix(test_y, predicted_y)

    # Recall matrix
    A = (((C.T) / (C.sum(axis=1))).T)

    # Precision matrix
    B = (C / C.sum(axis=0))

    plt.figure(figsize=(20, 4))

    labels = ['Re-paid(0)', 'Not Re-paid(1)']
    cmap = sns.light_palette("purple")
    plt.subplot(1, 3, 1)
    sns.heatmap(C, annot=True, cmap=cmap, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Orignal Class')
    plt.title('Confusion matrix')

    plt.subplot(1, 3, 2)
    sns.heatmap(A, annot=True, cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Orignal Class')
    plt.title('Recall matrix')

    plt.subplot(1, 3, 3)
    sns.heatmap(B, annot=True, cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Orignal Class')
    plt.title('Precision matrix')

    plt.show()


def cv_plot(alpha, cv_auc):
    fig, ax = plt.subplots()
    ax.plot(np.log10(alpha), cv_auc, c='g')
    for i, txt in enumerate(np.round(cv_auc, 3)):
        ax.annotate((alpha[i], str(txt)), (np.log10(alpha[i]), cv_auc[i]))
    plt.grid()
    plt.xticks(np.log10(alpha))
    plt.title("Cross Validation Error for each alpha")
    plt.xlabel("Alpha i's")
    plt.ylabel("Error measure")
    plt.show()


#%% Logistic Regression
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from sklearn.metrics import roc_auc_score

alpha = np.logspace(-6,6,9)
cv_auc_score = []
for i in alpha:
    clf = SGDClassifier(alpha=i, penalty='l1',class_weight = 'balanced', loss='log', random_state=28)
    clf.fit(X_train_final, y_train)
    sig_clf = CalibratedClassifierCV(clf, method='sigmoid')
    sig_clf.fit(X_train_final, y_train)
    y_pred_prob = sig_clf.predict_proba(X_val_final)[:,1]
    cv_auc_score.append(roc_auc_score(y_val,y_pred_prob))
    print('For alpha {0}, cross validation AUC score {1}'.format(i,roc_auc_score(y_val,y_pred_prob)))
cv_plot(alpha, cv_auc_score)
print('The Optimal C value is:', alpha[np.argmax(cv_auc_score)])

#%% Recall-Precision Matrix

from sklearn.metrics import accuracy_score

best_alpha = alpha[np.argmax(cv_auc_score)]
logreg = SGDClassifier(alpha = best_alpha, class_weight = 'balanced', penalty = 'l1', loss='log', random_state = 28)
logreg.fit(X_train_final, y_train)
logreg_sig_clf = CalibratedClassifierCV(logreg, method='sigmoid')
logreg_sig_clf.fit(X_train_final, y_train)
y_pred_prob = logreg_sig_clf.predict_proba(X_train_final)[:,1]
print('For best alpha {0}, The Train AUC score is {1}'.format(best_alpha, roc_auc_score(y_train,y_pred_prob) ))
y_pred_prob = logreg_sig_clf.predict_proba(X_val_final)[:,1]
print('For best alpha {0}, The Cross validated AUC score is {1}'.format(best_alpha, roc_auc_score(y_val,y_pred_prob) ))
y_pred_prob = logreg_sig_clf.predict_proba(X_test_final)[:,1]
print('For best alpha {0}, The Test AUC score is {1}'.format(best_alpha, roc_auc_score(y_test,y_pred_prob) ))
y_pred = logreg.predict(X_test_final)
print('The test AUC score is :', roc_auc_score(y_test,y_pred_prob))
print('The percentage of misclassified points {:05.2f}% :'.format((1-accuracy_score(y_test, y_pred))*100))
plot_confusion_matrix(y_test, y_pred)

#%% ROC-AUC

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc = roc_auc_score(y_test,y_pred_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, marker='.')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('ROC curve', fontsize = 20)
plt.xlabel('FPR', fontsize=15)
plt.ylabel('TPR', fontsize=15)
plt.grid()
plt.legend(["AUC=%.3f"%auc])
plt.show()


