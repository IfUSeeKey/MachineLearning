import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv("data.csv")

# 2.1

#   Базовая информация о выборке  
print(f"Кол-во объектов: {df.shape[0]}")
print(f"Кол-во признаков: {df.shape[1]}")
print("\nТипы данных:")
print(df.dtypes.value_counts())

#   Баланс классов  
print("\nРаспределение целевой переменной target:")
print(df['target'].value_counts())
print(f"Доля положительного класса: {df['target'].mean():.4f}")

#   Пропущенные значения  
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).sort_values(ascending=False)
missing_info = pd.DataFrame({
    'Пропуски': missing,
    'Процент': missing_pct
})
print("\nПризнаки с пропущенными значениями (топ-20):")
print(missing_info[missing_info['Пропуски'] > 0].head(20))
print(f"\nВсего признаков с пропусками: {(missing > 0).sum()}")

#   Дубликаты  
print(f"\nДоля дубликатов: {df.duplicated().mean():.4f}")
print(f"Кол-во дубликатов: {df.duplicated().sum()}")

#   Выбросы (визуально + IQR)  
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nКол-во числовых признаков: {len(numeric_cols)}")

# Ключевые признаки для анализа
key_cols = ['income', 'loan_body', 'annuity_payment', 'goods_price', 'days_birth', 'days_employed']
print("\nОписательные статистики ключевых признаков:")
print(df[key_cols].describe().T)

# Функция для подсчёта выбросов (определена ОТДЕЛЬНО, не внутри цикла!)
def count_outliers(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return ((series < lower) | (series > upper)).sum()

# Визуализация выбросов через boxplot
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.ravel(), key_cols):
    ax.boxplot(df[col].dropna(), vert=True)
    ax.set_title(col)
plt.tight_layout()
plt.savefig("boxplots.png", dpi=150)

# Подсчёт выбросов по правилу IQR
print("\nКол-во выбросов (IQR) по ключевым признакам:")
for col in key_cols:
    print(f"{col}: {count_outliers(df[col].dropna())}")



# 2.2
#   1. Удаляем нечисловые признаки  
non_numeric = [c for c in df.columns if df[c].dtype.name not in ['float64', 'int64']]
print("Удаляемые нечисловые признаки:", non_numeric)
df = df.drop(columns=non_numeric)

#   2. Удаляем идентификатор  
if 'reco_id_curr' in df.columns:
    df = df.drop(columns=['reco_id_curr'])

#   3. Удаляем дубликаты  
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Удалено дубликатов: {before - len(df)}")

#   4. Устранение выбросов  
# Для признаков с сильными выбросами применим клиппинг (обрезание по квантилям 1% и 99%)
# Это более безопасно, чем удаление строк (не теряем данные)
outlier_cols = ['income', 'loan_body', 'annuity_payment', 'goods_price',
                'days_employed']

for col in outlier_cols:
    if col in df.columns:
        q_low = df[col].quantile(0.01)
        q_high = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=q_low, upper=q_high)

print("Выбросы обрезаны по квантилям 1%/99%")

#   5. Заполнение пропущенных значений  
# Числовые признаки заполняем медианой (устойчива к выбросам)
missing_cols = df.columns[df.isnull().any()].tolist()
print(f"\nПризнаки с пропусками ({len(missing_cols)}):")
for col in missing_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

print(f"Пропусков после обработки: {df.isnull().sum().sum()}")
print(f"Итоговый размер: {df.shape}")

# 2.3

# 1. Отделяем целевую переменную (target) от признаков (X)
X = df.drop(columns=['target'])
y = df['target']

# 2. Разбиение 70/30 со стратификацией
# stratify=y критически важен при дисбалансе (у нас 8% положительных классов),
# чтобы доля целевого класса в train и test была одинаковой (~8%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y
)

print(f"Размер Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Баланс классов в train: {y_train.mean():.4f}")
print(f"Баланс классов в test:  {y_test.mean():.4f}")

# 3. Стандартизация (масштабирование)
scaler = StandardScaler()

# Обучаем scaler на тренировочных данных и сразу преобразуем их
X_train_scaled = scaler.fit_transform(X_train)

# Преобразуем тестовые данные, используя параметры (среднее и стд. отклонение), 
# полученные на train. 
# ВАЖНО: делаем только transform, а не fit_transform, чтобы избежать "утечки данных" (data leakage)
X_test_scaled = scaler.transform(X_test)

print("Данные успешно стандартизированы!")


# 2.4


class LogisticRegressionFromScratch:
    """
    Логистическая регрессия, реализованная с нуля на numpy.
    Обучение: градиентный спуск по логарифмической функции потерь (log-loss).
    """

    def __init__(self, learning_rate=0.1, n_iters=1000, tol=1e-6,
                 l2_reg=0.0, verbose=False):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.tol = tol
        self.l2_reg = l2_reg
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.loss_history = []

    @staticmethod
    def _sigmoid(z):
        # Численно устойчивая сигмоида (ограничиваем z, чтобы не было переполнения)
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_loss(self, y, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        # Бинарная кросс-энтропия (log-loss)
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        # L2-регуляризация (штраф за большие веса)
        loss += 0.5 * self.l2_reg * np.sum(self.weights ** 2)
        return loss

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0

        prev_loss = np.inf

        for i in range(self.n_iters):
            # Forward pass: вычисляем предсказания
            z = X @ self.weights + self.bias
            y_pred = self._sigmoid(z)

            # Вычисляем градиенты
            error = y_pred - y  # (n_samples, 1)
            dw = (X.T @ error) / n_samples + self.l2_reg * self.weights
            db = np.mean(error)

            # Обновление параметров (градиентный спуск)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Логирование функции потерь
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)

            if self.verbose and (i % 100 == 0 or i == self.n_iters - 1):
                print(f"Итерация {i:4d} | loss = {loss:.6f}")

            # Критерий ранней остановки (если loss почти не меняется)
            if abs(prev_loss - loss) < self.tol:
                if self.verbose:
                    print(f"Остановка на итерации {i} (сходимость)")
                break
            prev_loss = loss

        return self

    def predict_proba(self, X):
        """Возвращает вероятности принадлежности к классу 1"""
        z = X @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        """Возвращает предсказанные классы (0 или 1)"""
        return (self.predict_proba(X) >= threshold).astype(int)


#   Подготовка данных для обучения  
# Преобразуем pandas DataFrame и Series в numpy массивы
# y_train нужно.reshape((n, 1)), чтобы форма совпадала с weights (n_features, 1)
X_train_np = X_train_scaled
y_train_np = y_train.values.reshape(-1, 1)

print(f"Форма X_train: {X_train_np.shape}")
print(f"Форма y_train: {y_train_np.shape}")

#   Обучение модели  
model = LogisticRegressionFromScratch(
    learning_rate=0.1,
    n_iters=2000,
    tol=1e-7,
    l2_reg=1e-4,
    verbose=True
)
model.fit(X_train_np, y_train_np)

#   Визуализация кривой обучения  
plt.figure(figsize=(10, 4))
plt.plot(model.loss_history)
plt.xlabel("Итерация")
plt.ylabel("Log-loss")
plt.title("Кривая обучения логистической регрессии")
plt.grid(True)
plt.savefig("learning_curve.png", dpi=150)

print("\nОбучение завершено!")
print(f"Финальное значение loss: {model.loss_history[-1]:.6f}")


# 2.5

#   Подготовка данных для теста  
X_test_np = X_test_scaled
y_test_np = y_test.values

#   Предсказания  
# Для train
y_pred_train = model.predict(X_train_np).ravel()
y_proba_train = model.predict_proba(X_train_np).ravel()

# Для test
y_pred_test = model.predict(X_test_np).ravel()
y_proba_test = model.predict_proba(X_test_np).ravel()

#   Функция для вывода всех метрик  
def print_metrics(y_true, y_pred, y_proba, name):
    print("\n===== {} =====".format(name))
    print("Accuracy : {:.4f}".format(accuracy_score(y_true, y_pred)))
    print("Precision: {:.4f}".format(precision_score(y_true, y_pred, zero_division=0)))
    print("Recall   : {:.4f}".format(recall_score(y_true, y_pred, zero_division=0)))
    print("F1-score : {:.4f}".format(f1_score(y_true, y_pred, zero_division=0)))
    print("ROC AUC  : {:.4f}".format(roc_auc_score(y_true, y_proba)))
    
    print("\nМатрица ошибок (Confusion Matrix):")
    print(confusion_matrix(y_true, y_pred))
    
    print("\nПолный отчет классификации:")
    print(classification_report(y_true, y_pred, zero_division=0))

#   Вызов функции для обеих выборок  
print_metrics(y_train.values, y_pred_train, y_proba_train, "Обучающая выборка (Train)")
print_metrics(y_test.values, y_pred_test, y_proba_test, "Тестовая выборка (Test)")
