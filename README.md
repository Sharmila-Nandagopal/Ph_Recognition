    import pandas as pd
		import numpy as np
		import seaborn as sns
		from matplotlib import pyplot as plt
		data=pd.read_csv(r"C:\Users\user\Downloads\ph-data.csv")
		data.head()
		data.isnull().sum()
		X= data.iloc[:,:3]
		Y=data.iloc[:,3]
		X.isnull().sum()
		X.info()
		X.shape
		Y.shape
		from sklearn.model_selection import train_test_split
		train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size= 0.2, random_state= 100)
		train_x.shape
		train_y.shape
		test_x.shape
		test_y.shape
		from sklearn.linear_model import LogisticRegression
		model= LogisticRegression()
		model.fit(train_x,train_y)
		Log_pred= model.predict(test_x)
		Log_pred = pd.Series(Log_pred)
		Log_pred.head()
		data_fin= test_x
		data_fin['ActualLabel']= test_y
		data_fin['LogisticPred'] = Log_pred
		data_fin.to_csv(r"C:\Users\user\Downloads\ph-data_final.csv")
		data_fin.dropna(inplace=True)
		data_fin.head()
		from sklearn.metrics import classification_report
		print(classification_report(Log_pred, test_y))
		sns.displot(data_fin, x='LogisticPred')
		sns.lmplot(x = 'ActualLabel', y = 'LogisticPred', data=data_fin)
		sns.barplot(x = 'ActualLabel', y = 'LogisticPred', data=data_fin)
		dummy_X=pd.get_dummies(X)
		dummy_X.info()
		Y = data['label']
		train_X, test_X, train_y, test_y = train_test_split(dummy_X, Y, test_size = 0.2, random_state= 100)
		test_X.shape
		from sklearn.neighbors import KNeighborsClassifier
		model = KNeighborsClassifier()
		model.fit(train_X,train_y)
		KN_pred= model.predict(test_X)
		print(classification_report(test_y,KN_pred))
		KN_pred = pd.Series(KN_pred)
		KN_pred.head()
		data_fin['KNPred'] = KN_pred
		data_fin.head()
		sns.displot(data_fin, x = 'KNPred')
		sns.lmplot(x = 'ActualLabel', y = 'KNPred', data=data_fin)
		sns.barplot(x = 'ActualLabel', y = 'KNPred', data=data_fin)
		from sklearn.model_selection import cross_validate
		model = LogisticRegression()
		cross_validate(model,X,Y, cv=5,verbose= 2, n_jobs=-1)
		from sklearn.model_selection import GridSearchCV
		param={"n_neighbors":[5,10,15]}
		model=KNeighborsClassifier()
		gridcv= GridSearchCV(model,param, cv=5, verbose=2, n_jobs=-1)
		gridcv.fit(dummy_X,Y)
		gridcv.best_params_
		gridcv.best_estimator_
		gridcv.predict(test_X)
		gridy_pred=gridcv.predict(test_X)
		print(classification_report(test_y,gridy_pred))
		gridy_pred = pd.Series(gridy_pred)
		gridy_pred.head()
		data_fin['GridPred'] = gridy_pred
		data_fin.head()
		sns.barplot(x = 'ActualLabel', y = 'GridPred', data=data_fin)
		from sklearn.tree import DecisionTreeClassifier
		model= DecisionTreeClassifier()
		model.fit(train_X,train_y)
		Tree_pred = model.predict(test_X)
		print (classification_report(test_y,Tree_pred))
		Tree_pred = pd.Series(Tree_pred)
		Tree_pred.head()
		data_fin['TreePred'] = Tree_pred
		data_fin.head()
		sns.barplot(x = 'ActualLabel', y = 'TreePred', data=data_fin)
		from sklearn.ensemble import AdaBoostClassifier
		model = DecisionTreeClassifier()
		model_ada= AdaBoostClassifier(base_estimator =model, n_estimators = 2, learning_rate=0.1, random_state=None)
		model_ada.fit(train_X,train_y)
		Ada_pred=model_ada.predict(test_X)
		print(classification_report(test_y,Ada_pred))
		Ada_pred = pd.Series(Ada_pred)
		Ada_pred.head()
		data_fin['AdaPred']= Ada_pred
		data_fin.head()
		sns.barplot(x = 'ActualLabel', y = 'AdaPred', data=data_fin)
		from sklearn.ensemble import RandomForestClassifier
		param_ada={"n_estimators":[6,12,28,24], "learning_rate":[0.3,0.2,0.1]}
		model= AdaBoostClassifier()
		gridsearchcv = GridSearchCV(model,param_ada,verbose=2, cv=3, n_jobs=-1)
		gridsearchcv.fit(train_X, train_y)
		gridsearchcv.best_estimator_
		predy = gridsearchcv.predict(test_X)
		print (classification_report(test_y,predy))
		from sklearn.ensemble import GradientBoostingClassifier
		GBC_model = GradientBoostingClassifier()
		parameter = {"n_estimators":[5,10,25,20],"learning_rate":[0.1,0.3,0.5],"max_depth":[3,4,5,6,7],"min_samples_split":[30,40,50,60],"min_samples_leaf":[15,30,45,60,75]}
		grid_grad = GridSearchCV(GBC_model,parameter,verbose=2, cv=3, n_jobs=-1)
		grid_grad.fit(train_X, train_y)
		grid_grad.best_params_
		pred_grad=grid_grad.predict(test_X)
		print(classification_report(test_y, pred_grad))
		pred_grad = pd.Series(pred_grad)
		pred_grad.head()
		data_fin['GradPred'] = pred_grad
		data_fin.head()
		sns.barplot(x = 'ActualLabel', y = 'GradPred', data=data_fin)
		data_fin.head()
