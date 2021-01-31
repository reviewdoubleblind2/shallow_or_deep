# BASIC MODULES FOR THIS STEP
import optuna
import pandas as pd 
import numpy as np  
# READ AND PLOT 
import os 
import matplotlib.pyplot as plt
import seaborn as sns
# LOAD MODULES 
import joblib
# SPECIFIC MODELS FOR SCORES 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Approach multi or binary
approach = 'multi'
# Dataset Russell (R-DS), Juliet (J-DS) and OUR (GH-DS)
dataset= 'Juliet'
# Representation b0 (R0), b1 (R1), b1_int (R2), b1_iden (R3), b1_int_iden (R4)
rep= 'b1_int_iden'
# The path where the STUDY  (optimization process) is located 
basepath= ''
#Model  can be RF
NN= 'RF'
mode = 'tfidf'

# retrieve study 
study_name = '{}_{}_{}_{}'.format(NN, approach, dataset, rep) # Unique identifier of the study.
study = optuna.create_study(load_if_exists= True,direction="maximize", study_name=study_name, storage='sqlite:///{}_{}_{}_{}.db'.format(NN, approach, dataset, rep))

# get best trial
best_trial = study.best_trial
number = best_trial.number

#  best model
new_model = joblib.load( f'{rep}_{approach}_{number}.joblib')
params = best_trial.params

# READ X
X_train = np.load(os.path.join(basepath,'BOW',f'{approach}_{dataset}_{mode}_train_{rep}.npy'))
X_val = np.load(os.path.join(basepath,'BOW',f'{approach}_{dataset}_{mode}_val_{rep}.npy'))
X_test = np.load(os.path.join(basepath,'BOW',f'{approach}_{dataset}_{mode}_test_{rep}.npy'))

if approach == 'binary':
    Y_train = pd.read_csv(os.path.join(basepath,'Data_train_test',f'{approach}_Ytrain_{dataset}_{rep}.csv'))['VULN_N']
    Y_test = pd.read_csv(os.path.join(basepath,'Data_train_test',f'{approach}_Ytest_{dataset}_{rep}.csv'))['VULN_N']
    Y_val = pd.read_csv(os.path.join(basepath,'Data_train_test',f'{approach}_Yval_{dataset}_{rep}.csv'))['VULN_N']

else:
    Y_train = pd.read_csv(os.path.join(basepath,'Data_train_test',f'{approach}_Ytrain_{dataset}_{rep}.csv'))['TYPE_N']
    Y_test = pd.read_csv(os.path.join(basepath,'Data_train_test',f'{approach}_Ytest_{dataset}_{rep}.csv'))['TYPE_N']
    Y_val = pd.read_csv(os.path.join(basepath,'Data_train_test',f'{approach}_Yval_{dataset}_{rep}.csv'))['TYPE_N']


# TRAIN
predicted_train = new_model.predict(X_train)
# VALIDATION
predicted_val = new_model.predict(X_val)
# TEST
predicted_test = new_model.predict(X_test)

if approach == 'multi':

    # CLASSIFICATION REPORTS 
    labels =  list(Y_train.unique())
    report_train = classification_report(Y_train, predicted_train, labels = labels)
    report_val = classification_report(Y_val, predicted_val, labels = labels)
    report_test = classification_report(Y_test, predicted_test, labels = labels)

elif approach == 'binary':

    # CLASSIFICATION REPORTS 
    report_train = classification_report(Y_train, predicted_train, labels = ['VULN','NO_VULN'])
    report_val = classification_report(Y_val, predicted_val, labels = ['VULN','NO_VULN'])
    report_test = classification_report(Y_test, predicted_test, labels = ['VULN','NO_VULN'])

print('//'*5, 'TRAIN')

print("*"*5, 'Classification report','*'*5)
print(report_train)

#import pdb; pdb.set_trace()

print('*'*5, 'Confusion_matrix and Save plots', '*'*5)
cm = confusion_matrix(Y_train, predicted_train)
f = sns.heatmap(cm,fmt='g',  annot=True, cmap="Blues")
plt.xlabel("Predicted labels") 
plt.ylabel("True labels")
fig = f.get_figure()
fig.savefig(f'Train_{dataset}_{rep}_{approach}.png')

print("1",cm)
plt.clf()

cm = confusion_matrix(Y_train, predicted_train, normalize = 'true')
f = sns.heatmap(cm, fmt='.2f', annot=True, cmap="Blues")
plt.xlabel("Predicted labels") 
plt.ylabel("True labels")
fig = f.get_figure()
fig.savefig(f'Train_{dataset}_{rep}_{approach}_N.png')
plt.clf()

print('//'*5, 'VAL')
print("*"*5, 'Classification report','*'*5)
print(report_val)

#import pdb; pdb.set_trace()

print('*'*5, 'Confusion_matrix and Save plots', '*'*5)
plt.clf()

cm = confusion_matrix(Y_val, predicted_val)
f = sns.heatmap(cm,fmt='g',  annot=True, cmap="Blues")
plt.xlabel("Predicted labels") 
plt.ylabel("True labels")
fig = f.get_figure()
fig.savefig(f'Val_{dataset}_{rep}_{approach}.png')
print("2",cm)
plt.clf()

cm = confusion_matrix(Y_val, predicted_val, normalize = 'true')
f = sns.heatmap(cm, fmt='.2f', annot=True, cmap="Blues")
plt.xlabel("Predicted labels") 
plt.ylabel("True labels")
fig = f.get_figure()
fig.savefig(f'Val_{dataset}_{rep}_{approach}_N.png')


print('//'*5, 'TEST')
print("*"*5, 'Classification report','*'*5)
print(report_test)
#import pdb; pdb.set_trace()

print('*'*5, 'Confusion_matrix and Save plots', '*'*5)
plt.clf()
cm = confusion_matrix(Y_test, predicted_test)
f = sns.heatmap(cm,fmt='g',  annot=True, cmap="Blues")
plt.xlabel("Predicted labels") 
plt.ylabel("True labels")
fig = f.get_figure()
fig.savefig(f'Test_{dataset}_{rep}_{approach}.png')
print("3", cm)
plt.clf()

cm = confusion_matrix(Y_test, predicted_test, normalize = 'true')
f = sns.heatmap(cm, fmt='.2f', annot=True, cmap="Blues")
plt.xlabel("Predicted labels") 
plt.ylabel("True labels")
fig = f.get_figure()
fig.savefig(f'Test_{dataset}_{rep}_{approach}_N.png')
plt.clf()


print("*"*5,"Params","*"*5)
print(params)

# YOU COULD USE IT TO EVALUATE AND ANALYZE ALL THE TRIALS 
df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
