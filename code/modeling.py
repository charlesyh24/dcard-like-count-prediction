import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

def rf_model(X_train, y_train):
    """Random Forest model"""
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [200],
        'max_depth': [10],
        'min_samples_split': [5],
        'min_samples_leaf': [2],
    }
    
    rf = grid_search(rf, param_grid, X_train, y_train)
    
    return rf

def gb_model(X_train, y_train):
    """Gradient Boosting model"""
    gb = GradientBoostingRegressor(random_state=42)
    param_grid = {
        'n_estimators': [200],
        'learning_rate': [0.1],
        'max_depth': [4],
    }
    
    gb = grid_search(gb, param_grid, X_train, y_train)
    
    return gb

def dt_model(X_train, y_train):
    """Decision Tree model"""
    dt = DecisionTreeRegressor(random_state=42)
    param_grid = {
        'max_depth': [10],
        'min_samples_split': [5],
        'min_samples_leaf': [2],
    }
    
    dt = grid_search(dt, param_grid, X_train, y_train)
    
    return dt

def grid_search(model, param_grid, X_train, y_train):
    """Find parameters with best performance"""
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    
    return best_model

def ensemble(X_train, y_train, rf_model, gb_model, dt_model):
    
    model = VotingRegressor(estimators=[(rf_model[0], rf_model[1]),
                                        (gb_model[0], gb_model[1]),
                                        (dt_model[0], dt_model[1])])
    model.fit(X_train, y_train)
    
    return model

def train_model(X_train, y_train):
    """Train tree-based models with ensemble learning"""
    rf = rf_model(X_train, y_train)
    gb = gb_model(X_train, y_train)
    dt = dt_model(X_train, y_train)
    
    model = ensemble(X_train, y_train, ['rf', rf], ['gb', gb], ['dt', dt])
    
    return model

def calculate_mape(y_test, y_pred):
    """Calculate MAPE"""
    y_test, y_pred = np.exp(y_test), np.exp(y_pred)
    mape = ((y_test - y_pred) / y_test).abs().mean() * 100
    
    return mape

def eval_model(model, X_test, y_test):
    """Evaluate model"""
    y_pred = model.predict(X_test)
    mape = calculate_mape(y_test, y_pred)
    
    return mape
    
def integrate_model(X_train, X_test, y_train, y_test):
    """Integrate train_model and eval_model functions"""
    model = train_model(X_train, y_train)
    mape = eval_model(model, X_test, y_test)
    
    return mape