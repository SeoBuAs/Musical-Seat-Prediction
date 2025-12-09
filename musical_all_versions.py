import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import (
    RandomForestRegressor, 
    AdaBoostRegressor, 
    BaggingRegressor, 
    ExtraTreesRegressor, 
    GradientBoostingRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
import shap

from imblearn.over_sampling import SMOTE, RandomOverSampler, BorderlineSMOTE


class Config:
    DATA_ROOT = './data'
    OUTPUT_ROOT = './results'
    
    DATA_FILES = {
        'bare': 'bare.csv',
        'gentleman': 'gentleman.csv',
        'hades': 'hades.csv',
        'salieri': 'salieri.csv',
        'versailles': 'versailles.csv'
    }
    
    RANDOM_STATE = 42
    TEST_SIZE = 0.3
    VALIDATION_SPLIT = 0.5
    N_FOLDS = 5
    
    FEATURE_COLUMNS = [
        'cast1', 'cast2', 'cast3', 'cast4', 
        'weekend', 'day', 'dc', 'evt', 'Month', 
        'Day', 'night', 'musical_bare', 'musical_gentleman', 'musical_hades',
        'musical_salieri', 'musical_versailles'
    ]
    
    TARGET_COLUMN = 'seat'
    
    SAMPLING_VERSIONS = {
        'normal': None,
        'smote': SMOTE(sampling_strategy='auto', random_state=RANDOM_STATE),
        'random': RandomOverSampler(random_state=RANDOM_STATE),
        'borderline_smote': BorderlineSMOTE(random_state=RANDOM_STATE)
    }


def load_and_preprocess_data(config):
    print("Loading data...")
    
    dataframes = {}
    for name, filename in config.DATA_FILES.items():
        filepath = os.path.join(config.DATA_ROOT, filename)
        df = pd.read_csv(filepath)
        dataframes[name] = df
    
    processed_dfs = []
    
    for name, df in dataframes.items():
        df[['Month', 'DayTime']] = df['date'].str.split('/', expand=True)
        df[['Day', 'Time']] = df['DayTime'].str.split('.', expand=True)
        df['night'] = df['Time'].apply(lambda x: 1 if x == '5' else 0)
        
        df[f'musical_{name}'] = 1
        
        if name == 'hades' and 'dc ' in df.columns:
            df['dc'] = df['dc ']
        
        processed_dfs.append(df)
    
    data = pd.concat(processed_dfs, ignore_index=True)
    data = data.fillna(0)
    
    print(f"Total data size: {data.shape}")
    
    stratify_cols = ['musical_bare', 'musical_gentleman', 'musical_hades',
                     'musical_salieri', 'musical_versailles', config.TARGET_COLUMN]
    data['stratify_col'] = data[stratify_cols].astype(str).agg('_'.join, axis=1)
    
    value_counts = data['stratify_col'].value_counts()
    rare_classes = value_counts[value_counts == 1].index
    rare_class_data = data[data['stratify_col'].isin(rare_classes)]
    
    rare_X = rare_class_data[config.FEATURE_COLUMNS + ['stratify_col']]
    rare_y = rare_class_data[config.TARGET_COLUMN]
    
    data = data[data['stratify_col'].map(value_counts) > 1]
    
    all_columns = config.FEATURE_COLUMNS + ['stratify_col']
    X = data[all_columns]
    y = data[config.TARGET_COLUMN]
    
    print("\nSplitting data...")
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE, 
        stratify=X['stratify_col']
    )
    
    rare_classes_temp = X_temp['stratify_col'].value_counts()
    rare_classes_temp = rare_classes_temp[rare_classes_temp == 1].index
    rare_X_temp = X_temp[X_temp['stratify_col'].isin(rare_classes_temp)]
    rare_y_temp = y_temp.loc[rare_X_temp.index]
    
    X_temp = X_temp[~X_temp['stratify_col'].isin(rare_classes_temp)]
    y_temp = y_temp.loc[X_temp.index]
    
    X_test, X_ex, y_test, y_ex = train_test_split(
        X_temp, y_temp,
        test_size=config.VALIDATION_SPLIT,
        random_state=config.RANDOM_STATE,
        stratify=X_temp['stratify_col']
    )
    
    X_test = pd.concat([X_test, rare_X_temp, rare_X])
    y_test = pd.concat([y_test, rare_y_temp, rare_y])
    
    X_train = X_train.drop(columns='stratify_col').reset_index(drop=True).astype(float)
    X_test = X_test.drop(columns='stratify_col').reset_index(drop=True).astype(float)
    X_ex = X_ex.drop(columns='stratify_col').reset_index(drop=True).astype(float)
    
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    y_ex = y_ex.reset_index(drop=True)
    
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}, Extra: {X_ex.shape}")
    
    return X_train, X_test, X_ex, y_train, y_test, y_ex


def apply_oversampling(X_train, y_train, sampling_method, version_name):
    if sampling_method is None:
        print(f"\n[{version_name}] No oversampling - Using original data")
        return X_train, y_train
    
    print(f"\n[{version_name}] Applying oversampling...")
    print(f"  Before: {X_train.shape}")
    
    X_resampled, y_resampled = sampling_method.fit_resample(X_train, y_train)
    
    print(f"  After: {X_resampled.shape}")
    print(f"  Increase: {X_resampled.shape[0] - X_train.shape[0]} samples")
    
    return X_resampled, y_resampled


def get_models_and_params():
    models = {
        'RandomForestRegressor': RandomForestRegressor(random_state=0),
        'AdaBoostRegressor': AdaBoostRegressor(random_state=0),
        'BaggingRegressor': BaggingRegressor(random_state=0),
        'ExtraTreesRegressor': ExtraTreesRegressor(random_state=0),
        'GradientBoostingRegressor': GradientBoostingRegressor(random_state=0),
        'MLPRegressor': MLPRegressor(random_state=0, max_iter=1000),
        'CatBoostRegressor': CatBoostRegressor(random_state=0, verbose=0),
        'LGBMRegressor': lgb.LGBMRegressor(random_state=0, verbose=-1),
        'XGBRegressor': xgb.XGBRegressor(random_state=0, verbosity=0),
    }
    
    param_grids = {
        'RandomForestRegressor': {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None]
        },
        'AdaBoostRegressor': {
            'n_estimators': [50, 100],
            'learning_rate': [1.0, 0.1]
        },
        'BaggingRegressor': {
            'n_estimators': [50, 100],
            'max_samples': [0.8, 1.0],
            'max_features': [0.8, 1.0]
        },
        'ExtraTreesRegressor': {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        },
        'GradientBoostingRegressor': {
            'n_estimators': [50, 100],
            'learning_rate': [0.1, 0.01],
            'max_depth': [3, 5]
        },
        'MLPRegressor': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['tanh', 'relu'],
            'alpha': [0.0001, 0.001]
        },
        'SGDRegressor': {
            'loss': ['squared_error', 'huber', 'epsilon_insensitive'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'max_iter': [1000, 2000]
        },
        'SVR': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        },
        'CatBoostRegressor': {
            'iterations': [500, 1000],
            'learning_rate': [0.01, 0.1],
            'depth': [6, 8],
            'verbose': [0]
        },
        'LGBMRegressor': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [5, 10],
            'verbose': [-1]
        },
        'XGBRegressor': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'verbosity': [0]
        }
    }
    
    return models, param_grids


def train_model_nested_cv(model_name, model, param_grid, X_train, y_train, config):
    print(f'\n{"="*50}')
    print(f'Training model: {model_name}')
    print(f'{"="*50}')
    
    outer_cv = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    outer_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train), 1):
        print(f'\nOuter Fold {fold_idx}/{config.N_FOLDS}')
        
        X_nested_train = X_train.iloc[train_idx]
        X_nested_val = X_train.iloc[val_idx]
        y_nested_train = y_train.iloc[train_idx]
        y_nested_val = y_train.iloc[val_idx]
        
        inner_cv = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        grid_search = GridSearchCV(
            estimator=model, 
            param_grid=param_grid, 
            cv=inner_cv, 
            scoring='r2',
            n_jobs=-1
        )
        
        grid_search.fit(X_nested_train, y_nested_train)
        
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        
        y_pred = best_model.predict(X_nested_val)
        r2 = r2_score(y_nested_val, y_pred)
        mse = mean_squared_error(y_nested_val, y_pred)
        mae = mean_absolute_error(y_nested_val, y_pred)
        
        outer_scores.append((best_params, r2, mse, mae))
        print(f'  R2: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}')
        print(f'  Best Params: {best_params}')
    
    avg_scores_per_param = {}
    for params, r2, mse, mae in outer_scores:
        param_key = str(params)
        if param_key not in avg_scores_per_param:
            avg_scores_per_param[param_key] = []
        avg_scores_per_param[param_key].append((r2, mse, mae))
    
    best_param_set = None
    best_avg_r2 = -float('inf')
    best_avg_mse = None
    best_avg_mae = None
    
    for param_key, scores in avg_scores_per_param.items():
        avg_r2 = np.mean([s[0] for s in scores])
        avg_mse = np.mean([s[1] for s in scores])
        avg_mae = np.mean([s[2] for s in scores])
        
        if avg_r2 > best_avg_r2:
            best_avg_r2 = avg_r2
            best_avg_mse = avg_mse
            best_avg_mae = avg_mae
            best_param_set = param_key
    
    print(f'\nBest parameters: {best_param_set}')
    print(f'Average Validation R2: {best_avg_r2:.4f}')
    
    best_params_dict = eval(best_param_set)
    model.set_params(**best_params_dict)
    model.fit(X_train, y_train)
    
    validation_scores = {
        'R2': best_avg_r2,
        'MSE': best_avg_mse,
        'MAE': best_avg_mae
    }
    
    return model, best_params_dict, validation_scores


def evaluate_model(model, X, y, split_name='Test'):
    y_pred = model.predict(X)
    
    metrics = {
        'R2': r2_score(y, y_pred),
        'MSE': mean_squared_error(y, y_pred),
        'MAE': mean_absolute_error(y, y_pred)
    }
    
    print(f'\n{split_name} Performance:')
    print(f'  R2:  {metrics["R2"]:.4f}')
    print(f'  MSE: {metrics["MSE"]:.4f}')
    print(f'  MAE: {metrics["MAE"]:.4f}')
    
    return metrics


def run_version(version_name, config):
    print("\n" + "="*70)
    print(f"Running version: {version_name.upper()}")
    print("="*70)
    
    X_train, X_test, X_ex, y_train, y_test, y_ex = load_and_preprocess_data(config)
    
    sampling_method = config.SAMPLING_VERSIONS[version_name]
    X_train_resampled, y_train_resampled = apply_oversampling(
        X_train, y_train, sampling_method, version_name
    )
    
    models, param_grids = get_models_and_params()
    
    results = []
    
    for model_name, model in models.items():
        
        trained_model, best_params, val_metrics = train_model_nested_cv(
            model_name, model, param_grids[model_name],
            X_train_resampled, y_train_resampled, config
        )
        
        results.append({
            'Version': version_name,
            'Model': model_name,
            'Split': 'Validation_K-FOLD',
            'Best_Parameters': str(best_params),
            'R2': val_metrics['R2'],
            'MSE': val_metrics['MSE'],
            'MAE': val_metrics['MAE']
        })
        
        test_metrics = evaluate_model(trained_model, X_test, y_test, 'Test')
        results.append({
            'Version': version_name,
            'Model': model_name,
            'Split': 'Test',
            'Best_Parameters': str(best_params),
            'R2': test_metrics['R2'],
            'MSE': test_metrics['MSE'],
            'MAE': test_metrics['MAE']
        })
        
        ex_metrics = evaluate_model(trained_model, X_ex, y_ex, 'Extra')
        results.append({
            'Version': version_name,
            'Model': model_name,
            'Split': 'Extra',
            'Best_Parameters': str(best_params),
            'R2': ex_metrics['R2'],
            'MSE': ex_metrics['MSE'],
            'MAE': ex_metrics['MAE']
        })
    
    results_df = pd.DataFrame(results)
    output_dir = os.path.join(config.OUTPUT_ROOT, version_name)
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f'performance_metrics_{version_name}.csv')
    results_df.to_csv(results_file, index=False)
    
    print("\n" + "="*70)
    print(f"Version {version_name.upper()} completed!")
    print(f"Results saved to: {results_file}")
    print("="*70)
    
    return results_df


def main(versions_to_run='all'):
    config = Config()
    
    print("="*70)
    print("Musical Data Analysis - Multi-Version Experiment")
    print("="*70)
    
    if versions_to_run == 'all':
        versions = list(config.SAMPLING_VERSIONS.keys())
    elif isinstance(versions_to_run, list):
        versions = versions_to_run
    else:
        versions = [versions_to_run]
    
    print(f"\nVersions to run: {versions}\n")
    
    all_results = []
    for version in versions:
        if version not in config.SAMPLING_VERSIONS:
            print(f"Warning: '{version}' is not a valid version. Skipping.")
            continue
        
        version_results = run_version(version, config)
        all_results.append(version_results)
    
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_file = os.path.join(config.OUTPUT_ROOT, 'combined_all_versions.csv')
        combined_results.to_csv(combined_file, index=False)
        
        print("\n" + "="*70)
        print("All experiments completed!")
        print(f"Combined results: {combined_file}")
        print("="*70)
        
        print("\n[Test R² Summary by Version]")
        summary = combined_results[combined_results['Split'] == 'Test'].pivot_table(
            index='Model', 
            columns='Version', 
            values='R2'
        )
        print(summary.to_string())


if __name__ == '__main__':
    main(versions_to_run='all')
