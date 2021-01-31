import optuna
import pandas as pd 
import numpy as np
import tensorflow as tf
import keras     
#from sklearn.metrics import plot_confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os 
import matplotlib.pyplot as plt
#from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns

#Approach multi or binary
APP = 'binary'
# Dataset Russell (R-DS), Juliet (J-DS) and OUR (GH-DS)
DATA_S= 'OUR'
# Representation b0 (R0), b1 (R1), b1_int (R2), b1_iden (R3), b1_int_iden (R4)
REP_D= 'b1'
#Model  can be CNN OR RNN
NN= 'CNN'
# The path where the STUDY  (optimization process) is located 
base_path = ''


#retrieve study 
study_name = '{}_{}_{}_{}'.format(NN, APP, DATA_S, REP_D) # Unique identifier of the study.
study = optuna.create_study(load_if_exists= True,direction="maximize", study_name=study_name, storage='sqlite:///{}_{}_{}_{}.db'.format(NN, APP, DATA_S, REP_D))

# get best trial

best_trial = study.best_trial
number = best_trial.number

new_model = keras.models.load_model(f'{REP_D}_{number}.h5')
params = best_trial.params

X_train = np.load(os.path.join(base_path,'Sequence_tokens',f'{APP}_{DATA_S}_train_{REP_D}.npy'))
X_val = np.load(os.path.join(base_path,'Sequence_tokens',f'{APP}_{DATA_S}_val_{REP_D}.npy'))
X_test = np.load(os.path.join(base_path,'Sequence_tokens',f'{APP}_{DATA_S}_test_{REP_D}.npy'))

if APP  == 'binary':
    df_Y_train = pd.read_csv(os.path.join(base_path,'Data_train_test',f'{APP}_Ytrain_{DATA_S}_{REP_D}.csv'))
    df_Y_train['VULN_N'] = df_Y_train['VULN_N'].apply(lambda x: 1 if x=='VULN' else 0)
    Y_train = df_Y_train['VULN_N']

    df_Y_val = pd.read_csv(os.path.join(base_path,'Data_train_test',f'{APP}_Yval_{DATA_S}_{REP_D}.csv'))
    df_Y_val['VULN_N'] = df_Y_val['VULN_N'].apply(lambda x: 1 if x=='VULN' else 0)
    Y_val = df_Y_val['VULN_N']

    df_Y_test = pd.read_csv(os.path.join(base_path,'Data_train_test',f'{APP}_Ytest_{DATA_S}_{REP_D}.csv'))
    df_Y_test['VULN_N'] = df_Y_test['VULN_N'].apply(lambda x: 1 if x=='VULN' else 0)
    Y_test = df_Y_test['VULN_N']

else:
    pass

predicted_train =  new_model.predict_classes(X_train)
predicted_val =  new_model.predict_classes(X_val)
predicted_test =  new_model.predict_classes(X_test)

print('//'*5, 'TRAIN')

print("*"*5, 'Classification report','*'*5)
report = classification_report(Y_train, predicted_train, labels = [1,0], target_names=['VULN','NO_VULN'])
print(report)

print('*'*5, 'Confusion_matrix and Save plots', '*'*5)
cm = confusion_matrix(Y_train, predicted_train)
f = sns.heatmap(cm,fmt='g',  annot=True, cmap="Blues")
plt.xlabel("Predicted labels") 
plt.ylabel("True labels")
fig = f.get_figure()
fig.savefig(f'Train_{DATA_S}_{REP_D}_{APP}.png')

print("1",cm)
plt.clf()

cm = confusion_matrix(Y_train, predicted_train, normalize = 'true')
f = sns.heatmap(cm, fmt='g', annot=True, cmap="Blues")
plt.xlabel("Predicted labels") 
plt.ylabel("True labels")
fig = f.get_figure()
fig.savefig(f'Train_{DATA_S}_{REP_D}_{APP}_N.png')
plt.clf()

print('//'*5, 'VAL')
print("*"*5, 'Classification report','*'*5)
report = classification_report(Y_val, predicted_val, labels = [1,0], target_names=['VULN','NO_VULN'])
print(report)

print('*'*5, 'Confusion_matrix and Save plots', '*'*5)
plt.clf()

cm = confusion_matrix(Y_val, predicted_val)
f = sns.heatmap(cm,fmt='g',  annot=True, cmap="Blues")
plt.xlabel("Predicted labels") 
plt.ylabel("True labels")
fig = f.get_figure()
fig.savefig(f'Val_{DATA_S}_{REP_D}_{APP}.png')
print("2",cm)
plt.clf()

cm = confusion_matrix(Y_val, predicted_val, normalize = 'true')
f = sns.heatmap(cm, fmt='g', annot=True, cmap="Blues")
plt.xlabel("Predicted labels") 
plt.ylabel("True labels")
fig = f.get_figure()
fig.savefig(f'Val_{DATA_S}_{REP_D}_{APP}_N.png')

print('//'*5, 'TEST')
print("*"*5, 'Classification report','*'*5)
report = classification_report(Y_test, predicted_test, labels = [1,0], target_names=['VULN','NO_VULN'])
print(report)

print('*'*5, 'Confusion_matrix and Save plots', '*'*5)
plt.clf()
cm = confusion_matrix(Y_test, predicted_test)
f = sns.heatmap(cm,fmt='g',  annot=True, cmap="Blues")
plt.xlabel("Predicted labels") 
plt.ylabel("True labels")
fig = f.get_figure()
fig.savefig(f'Test_{DATA_S}_{REP_D}_{APP}.png')
print("3", cm)
plt.clf()

cm = confusion_matrix(Y_test, predicted_test, normalize = 'true')
f = sns.heatmap(cm, fmt='g', annot=True, cmap="Blues")
plt.xlabel("Predicted labels") 
plt.ylabel("True labels")
fig = f.get_figure()
fig.savefig(f'Test_{DATA_S}_{REP_D}_{APP}_N.png')
plt.clf()



df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
