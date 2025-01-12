# ml_models.py

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_selection import SelectKBest, f_regression
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@st.cache_resource
def train_ml_model(
    df,
    target_column,
    model_type="ML (линейная регрессия)",
    poly_degree=2,
    n_estimators=100,
    features=None,
    param_search_method="Нет",
    auto_feature_selection=False
):
    """
    Обучает модель ML (линейная/полиномиальная регрессия/случайный лес/SVR).

    Args:
        df (pd.DataFrame): DataFrame с данными для обучения. Обязательно наличие столбцов 'Месяц' и target_column.
        target_column (str): Имя целевого столбца.
        model_type (str): Тип модели ("ML (линейная регрессия)", "ML (полиномиальная регрессия)", "ML (случайный лес)", "ML (SVR)").
        poly_degree (int): Степень полинома для полиномиальной регрессии.
        n_estimators (int): Количество деревьев для случайного леса.
        features (list): Список признаков для обучения. Если None, используется только "Месяц".
        param_search_method (str): Метод поиска гиперпараметров ("Нет", "GridSearchCV", "RandomizedSearchCV").
        auto_feature_selection (bool): Если True, то включается автоматический выбор признаков.

    Returns:
        sklearn.base.BaseEstimator: Обученная модель.

    Raises:
        ValueError: Если в DataFrame нет столбцов 'Месяц' или целевой переменной, или если тип модели не поддерживается.
        Exception: Любая другая ошибка в процессе обучения.
    """
    try:
        if "Месяц" not in df.columns or target_column not in df.columns:
            raise ValueError("Нет столбцов 'Месяц' и/или целевой переменной.")

        X = df[["Месяц"]].values if features is None else df[features].values # Выбор признаков
        y = df[target_column].values  # Выбор целевой переменной

        if auto_feature_selection and features is not None and len(features) > 1:
            selector = SelectKBest(score_func=f_regression, k=min(3, len(features)))
            X = selector.fit_transform(X, y) # Автоматический выбор признаков
            selected_features = [features[i] for i in selector.get_support(indices=True)]
            logging.info(f"Автоматический выбор признаков: {selected_features}")

        # Инициализируем модель (без XGBoost)
        if model_type == "ML (линейная регрессия)":
            model = LinearRegression() # Линейная регрессия
        elif model_type == "ML (полиномиальная регрессия)":
            model = make_pipeline(PolynomialFeatures(poly_degree), LinearRegression()) # Полиномиальная регрессия
        elif model_type == "ML (случайный лес)":
            model = RandomForestRegressor(random_state=42, n_estimators=n_estimators)  # Случайный лес
        elif model_type == "ML (SVR)":
            model = SVR() # SVR
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}") # Ошибка если тип модели не поддерживается

        # Параметрический поиск
        if param_search_method == "GridSearchCV":
            param_grid = get_param_grid(model_type) # Сетка параметров для GridSearchCV
            model = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0)  # GridSearchCV
        elif param_search_method == "RandomizedSearchCV":
            param_dist = get_param_dist(model_type) # Распределение параметров для RandomizedSearchCV
            model = RandomizedSearchCV(
                model, param_dist, n_iter=10, cv=5,
                scoring='neg_mean_squared_error', random_state=42, verbose=0
            ) # RandomizedSearchCV

        model.fit(X, y) # Обучение модели
        cv = KFold(n_splits=5, shuffle=True, random_state=42) # Кросс-валидация
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error") # Вычисление кросс-валидации
        rmse_scores = np.sqrt(-scores) # Вычисление RMSE
        logging.info(f"'{model_type}' обучена. Средняя RMSE: {rmse_scores.mean():.2f}")
        return model

    except Exception as e:
        logging.error(f"Ошибка при обучении ML-модели: {e}")
        raise


def get_param_grid(model_type):
    """
    Возвращает словарь сетки параметров для GridSearchCV.

    Args:
        model_type (str): Тип модели ("ML (случайный лес)", "ML (SVR)").

    Returns:
         dict: Словарь с сеткой параметров для GridSearchCV. Возвращает пустой словарь, если тип модели не поддерживается.
    """
    if model_type == "ML (случайный лес)":
        return {
            'n_estimators': [100, 200, 300, 400], # Количество деревьев
            'max_depth': [None, 5, 10, 15],  # Максимальная глубина дерева
            'min_samples_split': [2, 5, 10], # Минимальное количество выборок для разделения
            'min_samples_leaf': [1, 2, 4],  # Минимальное количество выборок в листе
        }
    elif model_type == "ML (SVR)":
        return {
            'C': [0.1, 1, 10], # Параметр регуляризации
            'kernel': ['linear', 'rbf', 'poly'],  # Ядро
            'gamma': ['scale', 'auto'] # Параметр гамма
        }
    return {}


def get_param_dist(model_type):
    """
    Возвращает словарь распределений параметров для RandomizedSearchCV.

    Args:
         model_type (str): Тип модели ("ML (случайный лес)", "ML (SVR)").

    Returns:
        dict: Словарь с распределениями параметров для RandomizedSearchCV. Возвращает пустой словарь, если тип модели не поддерживается.
    """
    if model_type == "ML (случайный лес)":
        return {
            'n_estimators': np.arange(100, 500, 50),  # Количество деревьев
            'max_depth': [None] + list(np.arange(5, 20, 5)), # Максимальная глубина дерева
            'min_samples_split': np.arange(2, 15, 2),  # Минимальное количество выборок для разделения
            'min_samples_leaf': np.arange(1, 10, 2), # Минимальное количество выборок в листе
        }
    elif model_type == "ML (SVR)":
        return {
            'C': np.logspace(-1, 2, 10), # Параметр регуляризации
            'kernel': ['linear', 'rbf', 'poly'], # Ядро
            'gamma': ['scale', 'auto'] + list(np.logspace(-3, 0, 5)) # Параметр гамма
        }
    return {}


def predict_with_model(
    model,
    df,
    future_months,
    features=None,
    auto_feature_selection=False
):
    """
    Делает прогноз с помощью обученной модели.

    Args:
        model (sklearn.base.BaseEstimator): Обученная ML модель.
        df (pd.DataFrame): DataFrame с историческими данными.
        future_months (list): Список месяцев для прогнозирования.
        features (list): Список признаков для прогноза. Если None, используется только "Месяц".
        auto_feature_selection (bool): Если True, то используется автоматический выбор признаков

    Returns:
        tuple: Кортеж (predictions, intervals), где predictions - это прогнозы, а intervals - доверительные интервалы (если модель их возвращает)
    """
    try:
        if features is None:
            X_future = np.array(future_months).reshape(-1, 1)  # Использование только месяцев
        else:
            last_data = df[features].iloc[-1].to_dict()  # Последняя строка исторических данных
            future_df = pd.DataFrame({'Месяц': future_months})  # Создание DataFrame для будущих месяцев
            for feature in features:
                if feature != 'Месяц':
                    future_df[feature] = last_data[feature] # Заполнение фич последними значениями

            if auto_feature_selection and features is not None and len(features) > 1:
                if "Доходы" not in df.columns:
                    raise ValueError("В DataFrame нет столбца 'Доходы' для вычисления фич.")
                X = df[features].values
                selector = SelectKBest(score_func=f_regression, k=min(3, len(features)))
                selector.fit(X, df["Доходы"].values) # Автоматический выбор признаков
                selected_features = [features[i] for i in selector.get_support(indices=True)]
                X_future = future_df[selected_features].values
            else:
                X_future = future_df[features].values # Использование указанных признаков

            if hasattr(model, "predict_interval"):
                predictions, intervals = model.predict_interval(X_future, alpha=0.05) # Прогноз с интервалами
                return predictions, intervals

            if isinstance(model, type(make_pipeline(PolynomialFeatures(), LinearRegression()))):
                if not isinstance(model[0], PolynomialFeatures):
                    return model.predict(X_future), None # Прогноз без интервалов для полиномиальной регрессии
                predictions = model.predict(X_future)
                return predictions, None

        return model.predict(X_future), None # Прогноз без интервалов

    except Exception as e:
        logging.error(f"Ошибка при прогнозировании: {e}")
        raise


@st.cache_resource
def load_ml_model(
    df,
    target_column,
    model_type="ML (линейная регрессия)",
    poly_degree=2,
    n_estimators=100,
    features=None,
    param_search_method="Нет",
    auto_feature_selection=False,
    uploaded_model_file=None
):
    """
    Загружает или обучает (если нет готовой) ML-модель.

     Args:
        df (pd.DataFrame): DataFrame с данными для обучения. Обязательно наличие столбцов 'Месяц' и target_column.
        target_column (str): Имя целевого столбца.
        model_type (str): Тип модели ("ML (линейная регрессия)", "ML (полиномиальная регрессия)", "ML (случайный лес)", "ML (SVR)").
        poly_degree (int): Степень полинома для полиномиальной регрессии.
        n_estimators (int): Количество деревьев для случайного леса.
        features (list): Список признаков для обучения.
        param_search_method (str): Метод поиска гиперпараметров ("Нет", "GridSearchCV", "RandomizedSearchCV").
        auto_feature_selection (bool): Если True, то включается автоматический выбор признаков.
        uploaded_model_file (file): Загруженный файл с ML-моделью (.pkl). Если None, модель загружается из файла (если есть) или обучается заново.

    Returns:
        sklearn.base.BaseEstimator: Обученная или загруженная ML модель, None в случае ошибки или если нет данных.

    Raises:
          Exception: При ошибках загрузки или обучения модели
    """
    try:
        logging_info_message = ("Загрузка модели из загруженного файла."
                                if uploaded_model_file
                                else f"Загрузка/обучение модели: '{model_type}'.")
        logging.info(logging_info_message) # Сообщение о типе загрузки

        model_filename = (
            f"ml_model_{model_type}_{poly_degree}_{n_estimators}_"
            f"{'_'.join(features or [])}_{param_search_method}_{auto_feature_selection}.pkl"
        )  # Имя файла для сохранения модели

        if uploaded_model_file is not None:
            return joblib.load(uploaded_model_file)  # Загрузка модели из файла

        if df is not None:
            if os.path.exists(model_filename):
                logging.info(f"Загрузка модели из файла: {model_filename}")
                model = joblib.load(model_filename)  # Загрузка из файла (если есть)
            else:
                logging.info(f"Обучение новой модели: {model_filename}")
                model = train_ml_model(
                    df,
                    target_column,
                    model_type,
                    poly_degree,
                    n_estimators,
                    features,
                    param_search_method,
                    auto_feature_selection
                )  # Обучение новой модели
                save_ml_model(model, model_filename) # Сохранение новой модели
            return model
        else:
            logging.warning("Нет данных для обучения ML-модели.")  # Сообщение
            return None

    except Exception as e:
        logging.error(f"Ошибка при загрузке/обучении: {e}")
        return None


def save_ml_model(model, filepath="ml_model.pkl"):
    """
    Сохраняет обученную модель на диск.

    Args:
        model (sklearn.base.BaseEstimator): Обученная ML модель.
        filepath (str): Путь для сохранения файла.
    """
    try:
        joblib.dump(model, filepath)
        logging.info(f"Модель сохранена в {filepath}.")  # Сообщение
    except Exception as e:
        logging.error(f"Ошибка сохранения модели: {e}") # Сообщение об ошибке


def prepare_ml_data(df, target_column):
    """
    Добавляет признаки Lag_1, Lag_2, Rolling_Mean_3 и Rolling_Mean_5 в DataFrame.

    Args:
        df (pd.DataFrame): Входной DataFrame.
        target_column (str): Название целевого столбца

    Returns:
        pd.DataFrame: DataFrame с добавленными признаками
    """
    try:
        df = df.copy()
        avg = df[target_column].mean() if target_column in df.columns else 0 # Среднее значение
        df["Lag_1"] = df[target_column].shift(1).fillna(avg)  # Задержка на 1 шаг
        df["Lag_2"] = df[target_column].shift(2).fillna(avg) # Задержка на 2 шага
        df["Rolling_Mean_3"] = df[target_column].rolling(3, min_periods=1).mean() # Скользящее среднее за 3 месяца
        df["Rolling_Mean_5"] = df[target_column].rolling(5, min_periods=1).mean() # Скользящее среднее за 5 месяцев
        logging.info("Данные для ML подготовлены.") # Сообщение
        return df
    except Exception as e:
        logging.error(f"Ошибка при подготовке данных для ML: {e}") # Сообщение об ошибке
        return df


def calculate_metrics(y_true, y_pred):
    """
    Вычисляет метрики качества модели (RMSE, R2, MAE).

    Args:
        y_true (np.array): Фактические значения целевой переменной.
        y_pred (np.array): Прогнозируемые значения целевой переменной.

    Returns:
        tuple: Кортеж (RMSE, R2, MAE).
        RMSE (float): Корень среднеквадратической ошибки
        R2 (float): Коэффициент детерминации
        MAE (float): Средняя абсолютная ошибка

    """
    try:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Корень среднеквадратической ошибки
        r2 = r2_score(y_true, y_pred) # Коэффициент детерминации
        mae = mean_absolute_error(y_true, y_pred)  # Средняя абсолютная ошибка
        return rmse, r2, mae
    except Exception as e:
        logging.error(f"Ошибка при расчете метрик: {e}") # Сообщение об ошибке
        return None, None, None