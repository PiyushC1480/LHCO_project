# # import uproot
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import GradientBoostingClassifier
# # import xgboost as xgb
# # from sklearn.metrics import accuracy_score, confusion_matrix
# # from sklearn.metrics import roc_auc_score, roc_curve, auc



# # """
# # Designing boosted decision tree for the classification of tau hadronic decay and hard QCD jets

# # variable to use :  theta_J and N_T initially to segment phase space and then use lambda_J, r_2 and tau_31 to analyze the subspaces using multi-variate analysis
# # files given  : signal (tauhadronic_out.root) and background (hardqcd_200k_minpt50_out.root)
# # """
# # variables = ["LambdaJ", "ecfr2", "tau31"] # 



# # def create_df(file_name, label):
# #     file = uproot.open(f"../dataset/{file_name}.root")
# #     tree_name = list(file.keys())[0]
# #     tree = file[tree_name]
# #     df = tree.arrays(library="pd")
# #     # df = tree.arrays(variables, library="pd")
# #     df["label"] = label
# #     return df


# # #create a parquet file from the root file
# # def write_parquet(file_name):
# #     file  = uproot.open(f"../dataset/{file_name}.root")
# #     print(file.keys())
# #     tree_name = list(file.keys())[0]
# #     tree = file[tree_name]
# #     df = tree.arrays(library="pd")
# #     df.to_parquet(f"../dataset/converted/converted_{file_name}.parquet", index=False)
# #     print(f"saved {file_name}.parquet")
# #     return 


# # if __name__ == "__main__":

# #     signal_file = "tauhadronic_out"
# #     background_file = "hardqcd_200k_minpt50_out2"

# #     # write_parquet(signal_file)
# #     # write_parquet(background_file)

# #     signal_df = create_df(signal_file, 1)
# #     background_df = create_df(background_file, 0)

# #     print(len(signal_df), len(background_df))

# #     # write_parquet(signal_file)
# #     # write_parquet(background_file)

# #     df = pd.concat([signal_df, background_df], axis=0)
# #     df = df.sample(frac=1).reset_index(drop=True)

# #     X = df[variables]
# #     y = df["label"]

# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #     print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# #     # Gradient Boosting Classifier
# #     clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=3, random_state=42)
# #     clf.fit(X_train, y_train)
# #     y_pred = clf.predict(X_test)
# #     print(f"Accuracy GB: {accuracy_score(y_test, y_pred)}")
# #     print(f"ROC AUC GB: {roc_auc_score(y_test, y_pred)}")
# #     # print(f"Confusion Matrix GB: \n{confusion_matrix(y_test, y_pred)}")

# #     # XGBoost Classifier
# #     xgb_clf = xgb.XGBClassifier( n_estimators=100,  
# #     learning_rate=0.01,
# #     max_depth=3, 
# #     eval_metric="logloss",
# #     )

# #     xgb_clf.fit(X_train, y_train)
# #     y_pred = xgb_clf.predict(X_test)
# #     print(f"Accuracy XGB: {accuracy_score(y_test, y_pred)}")
# #     print(f"ROC AUC XGB: {roc_auc_score(y_test, y_pred)}")
# #     # print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")




# # import uproot
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import GradientBoostingClassifier
# # from sklearn.metrics import roc_auc_score, roc_curve, auc



# # """
# # Designing boosted decision tree for the classification of tau hadronic decay and hard QCD jets

# # variable to use :  theta_J and N_T initially to segment phase space and then use lambda_J, r_2 and tau_31 to analyze the subspaces using multi-variate analysis
# # files given  : signal (tauhadronic_out.root) and background (hardqcd_200k_minpt50_out.root)
# # """
# # variables = ["thetaJ", "trackno", "LambdaJ", "ecfr2", "tau31"] # 



# # def create_df(file_name, label):
# #     file = uproot.open(f"../dataset/{file_name}.root")
# #     tree_name = list(file.keys())[0]
# #     tree = file[tree_name]
# #     df = tree.arrays(library="pd")
# #     df["label"] = label
# #     return df


# # #create a parquet file from the root file
# # def write_parquet(file_name):
# #     file  = uproot.open(f"../dataset/{file_name}.root")
# #     print(file.keys())
# #     tree_name = list(file.keys())[0]
# #     tree = file[tree_name]
# #     df = tree.arrays(library="pd")
# #     df.to_parquet(f"../dataset/converted/converted_{file_name}.parquet", index=False)
# #     print(f"saved {file_name}.parquet")
# #     return 


# # if __name__ == "__main__":

# #     signal_file = "tauhadronic1L_minpt50_out"
# #     background_file = "hardqcd_200k_minpt50_out2"

# #     # write_parquet(signal_file)
# #     # write_parquet(background_file)

# #     signal_df = create_df(signal_file, 1)
# #     background_df = create_df(background_file, 0)

# #     print(len(signal_df), len(background_df))

# #     # write_parquet(signal_file)
# #     write_parquet(background_file)

# #     df = pd.concat([signal_df, background_df], axis=0)
# #     df = df.sample(frac=1).reset_index(drop=True)

# #     X = df[variables]
# #     y = df["label"]

# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #     print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")




    


# # import uproot
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import GradientBoostingClassifier
# # from sklearn.metrics import roc_auc_score, roc_curve, auc

# # from sklearn.model_selection import train_test_split, GridSearchCV
# # from sklearn.metrics import accuracy_score, classification_report
# # import xgboost as xgb



# # """
# # Designing boosted decision tree for the classification of tau hadronic decay and hard QCD jets

# # variable to use :  theta_J and N_T initially to segment phase space and then use lambda_J, r_2 and tau_31 to analyze the subspaces using multi-variate analysis
# # files given  : signal (tauhadronic_out.root) and background (hardqcd_200k_minpt50_out.root)
# # """
# # variables = ["thetaJ", "trackno", "LambdaJ", "ecfr2", "tau31"] #



# # def create_df(file_name, label):
# #     file = uproot.open(f"/content/{file_name}.root")  # Corrected path
# #     tree_name = list(file.keys())[0]
# #     tree = file[tree_name]
# #     df = tree.arrays(library="pd")[variables]
# #     df["label"] = label
# #     return df

# # #create a parquet file from the root file
# # def write_parquet(file_name):
# #     file  = uproot.open(f"/content/{file_name}.root")  # Corrected path
# #     print(file.keys())
# #     tree_name = list(file.keys())[0]
# #     tree = file[tree_name]
# #     df = tree.arrays(library="pd")
# #     df.to_parquet(f"/content/converted_{file_name}.parquet", index=False)  # Save in /content/
# #     print(f"saved {file_name}.parquet")
# #     return


# # if __name__ == "__main__":

# #     signal_file = "tauhadronic1L_minpt50_out"
# #     background_file = "hardqcd_200k_minpt50_out2"

# #     # write_parquet(signal_file)
# #     # write_parquet(background_file)

# #     signal_df = create_df(signal_file, 1)
# #     background_df = create_df(background_file, 0)

# #     print(len(signal_df), len(background_df))

# #     # write_parquet(signal_file)
# #     write_parquet(background_file)

# #     df = pd.concat([signal_df, background_df], axis=0)
# #     df = df.sample(frac=1).reset_index(drop=True)

# #     X = df[variables]
# #     y = df["label"]

# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #     print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")






# import uproot
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import roc_auc_score, roc_curve, auc

# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import accuracy_score, classification_report
# import xgboost as xgb


# def create_df(file_name, label):
#     file = uproot.open(f"../dataset/{file_name}.root")  # Corrected path
#     tree_name = list(file.keys())[0]
#     tree = file[tree_name]
#     df = tree.arrays(library="pd")
#     df["label"] = label
#     return df


# signal_file = "tauhadronic_out"
# background_file = "hardqcd_200k_minpt50_out2"

# signal_df = create_df(signal_file, 1)
# background_df = create_df(background_file, 0)

# df = pd.concat([signal_df, background_df], axis=0)
# df = df.sample(frac=1).reset_index(drop=True)
# # Features and labels
# X = df.drop(columns=["label"])
# y = df["label"]

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Define XGBoost classifier
# xgb_clf = xgb.XGBClassifier( eval_metric="logloss")

# # Grid Search Hyperparameters
# param_grid = {
#     "n_estimators": [50, 100, 200],
#     "max_depth": [3, 5, 7],
#     "learning_rate": [0.01, 0.1, 0.2],
#     "subsample": [0.7, 0.8, 1.0]
# }

# # Grid Search with Cross-Validation
# grid_search = GridSearchCV(xgb_clf, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
# grid_search.fit(X_train, y_train)

# # Best model and predictions
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test)
# y_prob = best_model.predict_proba(X_test)[:, 1]

# # Evaluation
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Best Parameters: {grid_search.best_params_}")
# print(f"Accuracy: {accuracy:.4f}")
# print("Classification Report:\n", classification_report(y_test, y_pred))


# # Compute ROC curve and AUC
# fpr, tpr, _ = roc_curve(y_test, y_prob)
# roc_auc = roc_auc_score(y_test, y_prob)





# import uproot
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingClassifier
# import xgboost as xgb
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.metrics import roc_auc_score, roc_curve, auc
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import precision_score, recall_score, f1_score
# import sys

# sys.stdout = open("output_new2.txt", "w")


# """
# Designing boosted decision tree for the classification of tau hadronic decay and hard QCD jets

# variable to use :  theta_J and N_T initially to segment phase space and then use lambda_J, r_2 and tau_31 to analyze the subspaces using multi-variate analysis
# files given  : signal (tauhadronic_out.root) and background (hardqcd_200k_minpt50_out.root)
# """
# variables = ['trackno', 'thetaJ', 'LambdaJ', 'epsJ', 'tau1']
# # ['trackno', 'thetaJ', 'LambdaJ', 'ecrf2']


# def create_df(file_name, label):
#     file = uproot.open(f"../dataset/{file_name}.root")
#     tree_name = list(file.keys())[0]
#     tree = file[tree_name]
#     # df = tree.arrays(library="pd")
#     df = tree.arrays(variables, library="pd")
#     df["label"] = label
#     return df


# #create a parquet file from the root file
# def write_parquet(file_name):
#     file  = uproot.open(f"../dataset/{file_name}.root")
#     print(file.keys())
#     tree_name = list(file.keys())[0]
#     tree = file[tree_name]
#     df = tree.arrays(library="pd")
#     df.to_parquet(f"../dataset/converted/converted_{file_name}.parquet", index=False)
#     print(f"saved {file_name}.parquet")
#     return 


# if __name__ == "__main__":

#     signal_file = "tauhadronic1L_minpt50_out"
#     background_file = "hardqcd_200k_minpt50_out2"

#     # write_parquet(signal_file)
#     # write_parquet(background_file)

#     signal_df = create_df(signal_file, 1)
#     background_df = create_df(background_file, 0)

#     print(len(signal_df), len(background_df))

#     # write_parquet(signal_file)
#     # write_parquet(background_file)

#     df = pd.concat([signal_df, background_df], axis=0)
#     df = df.sample(frac=1).reset_index(drop=True)

#     X = df[variables]
#     y = df["label"]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

#     # Gradient Boosting Classifier
#     # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=3, random_state=42)
#     # clf.fit(X_train, y_train)
#     # y_pred = clf.predict(X_test)
#     # print(f"Accuracy GB: {accuracy_score(y_test, y_pred)}")
#     # print(f"ROC AUC GB: {roc_auc_score(y_test, y_pred)}")
#     # print(f"Confusion Matrix GB: \n{confusion_matrix(y_test, y_pred)}")

#     # clf = GradientBoostingClassifier(random_state=42)
#     param_grid = {
#         'n_estimators': [50, 100, 200],
#         'learning_rate': [0.01, 0.05, 0.1, 0.2],
#         'max_depth': [3, 5, 7],
#         "subsample": [0.7, 0.8, 1.0]
#     }
#     # # Perform Grid Search   
#     # grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
#     # grid_search.fit(X_train, y_train)

#     # # Best parameters and best model
#     # print(f"Best Parameters: {grid_search.best_params_}")
#     # best_model = grid_search.best_estimator_

#     # y_pred = best_model.predict(X_test)
#     # print(f"Accuracy GB: {accuracy_score(y_test, y_pred)}")
#     # print(f"ROC AUC GB: {roc_auc_score(y_test, y_pred)}")
#     # print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")
    
#     # print(f"Precision (Macro): {precision_score(y_test, y_pred, average='macro')}")
#     # print(f"Recall (Macro): {recall_score(y_test, y_pred, average='macro')}")
#     # print(f"F1-score (Macro): {f1_score(y_test, y_pred, average='macro')}")

#     # print(f"Precision (Micro): {precision_score(y_test, y_pred, average='micro')}")
#     # print(f"Recall (Micro): {recall_score(y_test, y_pred, average='micro')}")
#     # print(f"F1-score (Micro): {f1_score(y_test, y_pred, average='micro')}")

#     # XGBoost Classifier
#     xgb_clf = xgb.XGBClassifier(eval_metric="logloss")
#     grid_search2 = GridSearchCV(xgb_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

#     grid_search2.fit(X_train, y_train)

#     # Best parameters and best model
#     print(f"Best Parameters: {grid_search2.best_params_}")
#     best_model = grid_search2.best_estimator_

#     y_pred = best_model.predict(X_test)
#     print(f"Accuracy XGB: {accuracy_score(y_test, y_pred)}")
#     print(f"ROC AUC XGB: {roc_auc_score(y_test, y_pred)}")
#     print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")

#     print(f"Precision (Macro): {precision_score(y_test, y_pred, average='macro')}")
#     print(f"Recall (Macro): {recall_score(y_test, y_pred, average='macro')}")
#     print(f"F1-score (Macro): {f1_score(y_test, y_pred, average='macro')}")

#     print(f"Precision (Micro): {precision_score(y_test, y_pred, average='micro')}")
#     print(f"Recall (Micro): {recall_score(y_test, y_pred, average='micro')}")
#     print(f"F1-score (Micro): {f1_score(y_test, y_pred, average='micro')}")

#     # xgb_clf.fit(X_train, y_train)
#     # y_pred = xgb_clf.predict(X_test)
#     # print(f"Accuracy XGB: {accuracy_score(y_test, y_pred)}")
#     # print(f"ROC AUC XGB: {roc_auc_score(y_test, y_pred)}")
#     # print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")

#     sys.stdout.close()



# import h5py
# import pandas as pd
# import numpy as np


# def h5_to_csv(h5_file, csv_file, dataset_name="df"):
#     # Read the HDF5 file using pandas
#     df = pd.read_hdf(h5_file, key=dataset_name)
    
#     # Save as CSV
#     df.to_csv(csv_file, index=False)
#     print(f"Saved dataset '{dataset_name}' to {csv_file}")


# # Example usage
# in_file  = "../dataset/events_anomalydetection_Z_XY_qqq.h5"
# out_file = "../output/qqq.txt"

# h5_to_csv(in_file, out_file)
