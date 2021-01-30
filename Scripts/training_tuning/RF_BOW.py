import warnings

import numpy as np
import pandas as pd
import os
import joblib
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import optuna

#Approach multi or binary
approach = 'multi'
# Dataset Russell (R-DS), Juliet (J-DS) and OUR (GH-DS)
dataset= 'Russell'
# Representation b0 (R0), b1 (R1), b1_int (R2), b1_iden (R3), b1_int_iden (R4)
rep= 'b0'
# The path where the folder of the data (X and Y) is located
basepath= ''
#Model  can be RF
NN= 'RF'
mode = 'tfidf'

def objective(trial):

    approach = APP
    dataset = DATA_S
    rep = REP_D
    mode = 'tfidf'

    x_train = np.load(os.path.join(basepath,'BOW',f'{approach}_{dataset}_{mode}_train_{rep}.npy'))
    x_val = np.load(os.path.join(basepath,'BOW',f'{approach}_{dataset}_{mode}_val_{rep}.npy'))
    
    if approach == 'binary':
        y_train = pd.read_csv(os.path.join(basepath,'Data_train_test',f'{approach}_Ytrain_{dataset}_{rep}.csv'))['VULN_N']
        y_val = pd.read_csv(os.path.join(basepath,'Data_train_test',f'{approach}_Yval_{dataset}_{rep}.csv'))['VULN_N']

    else:
        y_train = pd.read_csv(os.path.join(basepath,'Data_train_test',f'{approach}_Ytrain_{dataset}_{rep}.csv'))['TYPE_N']
        y_val = pd.read_csv(os.path.join(basepath,'Data_train_test',f'{approach}_Yval_{dataset}_{rep}.csv'))['TYPE_N']
        
    criterion= trial.suggest_categorical("criterion",['gini','entropy'])
    estimators= trial.suggest_int('n_estimators', 5, 400)
    max_features= trial.suggest_categorical("max_features",['sqrt','log2'])
    max_depth= trial.suggest_categorical("max_depth",[10,15,30, 50, 100,None]) 

    random_forest = RandomForestClassifier(n_jobs=-1,class_weight= "balanced",random_state=0,\
                                n_estimators= estimators,max_features= max_features,\
                                max_depth= max_depth,criterion = criterion)

    random_forest.fit(x_train, y_train)
    y_pred = random_forest.predict(x_val)
    score = accuracy_score(y_val, y_pred)

    joblib.dump(random_forest, f'{rep}_{approach}_{trial.number}.joblib')

    return score


if __name__ == "__main__":
    warnings.warn(
        "Recent Keras release (2.4.0) simply redirects all APIs "
        "in the standalone keras package to point to tf.keras. "
    )


    study_name = '{}_{}_{}_{}'.format(NN, approach, dataset, rep) # Unique identifier of the study.

    study = optuna.create_study(load_if_exists= True,direction="maximize", study_name=study_name, storage='sqlite:///{}_{}_{}_{}.db'.format(NN, approach, dataset, rep))
    study.optimize(objective, n_trials=25)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))