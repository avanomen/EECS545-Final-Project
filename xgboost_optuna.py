import optuna
import pandas as pd
import numpy as np
import xgboost2
import pickle as pkl

# import dataframes
X_train = pd.read_csv('./data/X_train_adult.csv')
X_test = pd.read_csv('./data/X_test_adult.csv')
y_train = pd.read_csv('./data/y_train_adult.csv')
y_test = pd.read_csv('./data/y_test_adult.csv')

print(X_train.shape)
print(y_train.values.shape)
print(y_train.columns)

def objective(trial):
    reg = trial.suggest_float("reg", 1e-2, 1.0)
    gamma = trial.suggest_float("gamma", 1e-2, 3.0)
    feature_sel = trial.suggest_float("feature_sel", 0.2, 1.0)
    max_depth = trial.suggest_int("max_depth", 2, 20)
    min_child_weight = trial.suggest_float("min_child_weight", 1, 5)
    lr = trial.suggest_float("lr", 1e-2, 1.0, log=True)
    min_leaf_num = trial.suggest_int("min_leaf_num", 3, 20)
    boosting_rounds = trial.suggest_int("boosting_rounds", 2, 10, step=2)

    # train the model with the given parameters
    model = xgboost2.XGBoostClassifier()
    model.fit(X_train.values, y_train['income'].values,
              boosting_rounds=boosting_rounds, 
              feature_sel=feature_sel, 
              min_num_leaf=min_leaf_num, 
              min_child_weight=min_child_weight, 
              max_depth=max_depth, 
              lr=lr, 
              reg=reg, 
              gamma=gamma)
  
    pred = model.predict(X_test.values)
    acc = np.sum(pred == y_test['income'].values)/len(pred)

    return acc

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # save study object
    with open('adult_study.pkl', 'rb') as out_file:
        pkl.dump(study, out_file, pkl.HIGHEST_PROTOCOL)





