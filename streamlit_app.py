import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- КОНФИГУРАЦИЯ И КОНСТАНТЫ ---
RANDOM_STATE = 42
MAX_ORIGINAL_GRADE = 10.0
MAX_TARGET_GRADE = 100.0

# --- 1. ФУНКЦИИ ОБРАБОТКИ ДАННЫХ И МОДЕЛИРОВАНИЯ ---

@st.cache_data
def load_and_preprocess_data(uploaded_file):
    """Загрузка, предобработка и масштабирование данных."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Для тестового деплоя или если файл не загружен
        st.error("Пожалуйста, загрузите файл 'student_prediction.csv'.")
        return None, None, None, None

    # Проверка обязательных колонок
    required_cols = ['Student_ID', 'Student_Age', 'Sex', 'High_School_Type', 'Scholarship', 'Grade']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Файл должен содержать колонки: {', '.join(required_cols)}")
        return None, None, None, None

    # Масштабирование целевой переменной
    df['Grade_100'] = (df['Grade'] / MAX_ORIGINAL_GRADE) * MAX_TARGET_GRADE
    
    # Предварительная очистка 'Scholarship'
    df['Scholarship'] = df['Scholarship'].astype(str).str.replace('%', '', regex=False).astype(float)

    X = df.drop(['Student_ID', 'Grade', 'Grade_100'], axis=1)
    y = df['Grade_100']
    
    return df, X, y, df['Student_ID'] # Возвращаем df для отображения

def get_preprocessor():
    """Создание ColumnTransformer."""
    numerical_features = ['Student_Age', 'Weekly_Study_Hours', 'Scholarship']
    categorical_features = [
        'Sex', 'High_School_Type', 'Transportation', 'Attendance',
        'A6itional_Work', 'Sports_activity', 'Reading', 'Notes',
        'Listening_in_Class', 'Project_work'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ],
        remainder='drop'
    )
    return preprocessor

@st.cache_resource
def train_model(X_train, y_train, n_iter, max_depth, learning_rate):
    """Обучение модели с использованием RandomizedSearchCV."""
    preprocessor = get_preprocessor()
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=RANDOM_STATE, objective='reg:squarederror'))
    ])

    # Ограниченная сетка для быстрого деплоя
    param_distributions = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__learning_rate': [0.01, learning_rate],
        'regressor__max_depth': [3, max_depth, 7],
    }

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=n_iter,
        scoring='neg_mean_squared_error',
        cv=3, # Уменьшаем CV до 3 для скорости
        verbose=0,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    random_search.fit(X_train, y_train)
    return random_search.best_estimator_, random_search.best_params_

# --- 2. ИНТЕРФЕЙС STREAMLIT ---

st.set_page_config(layout="wide", page_title="Предсказание оценки студента (XGBoost)")

st.title("🎓 Предсказание оценки студента на 100-балльной шкале")

# --- Боковая панель для настройки ---
st.sidebar.header("⚙️ Настройки данных и модели")
uploaded_file = st.sidebar.file_uploader(
    "Загрузите CSV-файл (student_prediction.csv):", 
    type=["csv"]
)

st.sidebar.markdown("### Гиперпараметры XGBoost")
n_iter = st.sidebar.slider("Количество итераций поиска (n_iter)", 10, 100, 30, step=10)
max_depth = st.sidebar.slider("Максимальная глубина дерева (max_depth)", 3, 15, 5)
learning_rate = st.sidebar.slider("Скорость обучения (learning_rate)", 0.01, 0.2, 0.1, 0.01)
test_size = st.sidebar.slider("Размер тестовой выборки (%)", 10, 50, 20) / 100


if uploaded_file is not None:
    df, X, y, student_ids = load_and_preprocess_data(uploaded_file)
    
    if df is not None:
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE
        )
        
        # --- Основное поле ---
        
        st.header("1. Обзор данных")
        st.write(f"Размерность данных: {df.shape[0]} строк, {df.shape[1]} колонок.")
        st.dataframe(df.head())
        
        if st.button('🚀 Обучить и оценить модель'):
            with st.spinner('Обучение модели и поиск гиперпараметров...'):
                best_model, best_params = train_model(X_train, y_train, n_iter, max_depth, learning_rate)
            
            st.success("✅ Обучение и оптимизация завершены!")
            
            # --- Результаты обучения ---
            st.header("2. Результаты обучения и оценки")
            
            st.subheader("2.1. Лучшие параметры")
            st.json(best_params)

            # Оценка
            y_pred = best_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("MSE", f"{mse:.4f}")
            col2.metric("RMSE (Ошибка)", f"{rmse:.4f} балла")
            col3.metric("R-квадрат", f"{r2:.4f}")
            
            if r2 > 0.7:
                st.info(f"🎉 Модель показывает отличные результаты: R-квадрат > 0.7, что означает, что {r2*100:.2f}% дисперсии объяснено.")

            # --- Сравнение предсказаний ---
            st.subheader("2.2. Сравнение предсказаний")
            y_pred_rounded = np.round(y_pred, 2)
            y_test_rounded = np.round(y_test, 2)
            
            test_indices = X_test.index
            original_test_ids = df.loc[test_indices, 'Student_ID']

            results_df = pd.DataFrame({
                'Student_ID': original_test_ids,
                'Фактическая оценка (100)': y_test_rounded,
                'Предсказанная оценка (100)': y_pred_rounded,
                'Абс. Ошибка': np.abs(y_test_rounded - y_pred_rounded)
            }).reset_index(drop=True)
            
            tab1, tab2 = st.tabs(["Лучшие предсказания", "Худшие предсказания"])
            with tab1:
                st.dataframe(results_df.sort_values(by='Абс. Ошибка', ascending=True).head(10))
            with tab2:
                st.dataframe(results_df.sort_values(by='Абс. Ошибка', ascending=False).head(10))

            # --- Важность признаков ---
            st.header("3. Интерпретация: Важность признаков")
            
            # Получаем имена признаков
            feature_names_out = best_model['preprocessor'].get_feature_names_out()
            feature_names_model = [name.split('__')[-1] for name in feature_names_out]
            feature_importances = best_model['regressor'].feature_importances_
            importance_series = pd.Series(feature_importances, index=feature_names_model)
            importance_series = importance_series.sort_values(ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=importance_series.values, y=importance_series.index, palette="viridis", ax=ax)
            ax.set_title('Топ-10 наиболее важных признаков')
            ax.set_xlabel('Важность признака (Gain)')
            ax.set_ylabel('Признак')
            st.pyplot(fig)
            
            st.markdown("---")
            st.subheader("Сводка Топ-5")
            st.dataframe(importance_series.head(5))

else:
    st.info("⬆️ Пожалуйста, загрузите файл `student_prediction.csv` на боковой панели, чтобы начать.")
    st.markdown("### Пример ожидаемого формата данных:")
    st.dataframe({
        'Student_ID': ['STUDENT1', 'STUDENT2'],
        'Student_Age': [20, 18],
        'Sex': ['Male', 'Female'],
        # ... и все остальные колонки, включая 'Grade'
        'Grade': [9.0, 8.5]
    })
