def create_model(params, X_train, y_train, method = DecisionTreeRegressor(), cv= 5, scoring = "neg_mean_squared_error"):
    grid_search = GridSearchCV(estimator = method, param_grid=params, cv = cv, scoring = scoring, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search