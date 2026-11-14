import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Загрузка данных
df = pd.read_csv("data.csv")

# Первичный осмотр
print("Размер выборки:", df.shape)
print("\nПервые 5 строк:")
print(df.head())
print("\nИнформация о типах данных:")
print(df.info())
print("\nОписательная статистика:")
print(df.describe())
print("\nКоличество пропусков в каждой колонке:")
print(df.isnull().sum().sort_values(ascending=False).head(20))
print("\nРаспределение целевой переменной:")
print(df["target"].value_counts(normalize=True))

# ----------
# 1. Первичная глобальная очистка
# ----------

# Удаляем идентификатор
df = df.drop(columns=["reco_id_curr"])

# Удаляем колонки с большим количеством пропусков (более 50%)
threshold = len(df) * 0.5
cols_to_drop = df.columns[df.isnull().sum() > threshold]
print("\nКолонки с пропусками > 50%, будут удалены:")
print(cols_to_drop.tolist())
df = df.drop(columns=cols_to_drop)

# ----------
# 2. РАЗДЕЛЕНИЕ ДАННЫХ
# ----------

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("\nРазмер обучающей выборки:", X_train.shape)
print("Размер контрольной выборки:", X_test.shape)

# ----------
# 3. Предобработка ТОЛЬКО на основе статистики обучающей выборки
# ----------

print("\nЗаполнение пропусков статистикой из train...")

# Числовые признаки — медианой из X_train
numeric_cols = X_train.select_dtypes(include=["number"]).columns
medians = X_train[numeric_cols].median()

X_train[numeric_cols] = X_train[numeric_cols].fillna(medians)
X_test[numeric_cols] = X_test[numeric_cols].fillna(medians) # Применяем медианы train к test!

# Категориальные признаки — модой из X_train
categorical_cols = X_train.select_dtypes(include=["object", "string"]).columns

for col in categorical_cols:
    mode_val = X_train[col].mode()[0]
    X_train[col] = X_train[col].fillna(mode_val)
    X_test[col] = X_test[col].fillna(mode_val) # Применяем моду train к test!

print("Пропуски в train после обработки:", X_train.isnull().sum().sum())
print("Пропуски в test после обработки:", X_test.isnull().sum().sum())

# ----------
# 4. Кодирование категориальных признаков
# ----------

# Кодируем train и test отдельно
X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# В test могут не встретиться некоторые категории, которые были в train (или наоборот).
# Мы приводим X_test к точно такому же набору колонок, как в X_train, заполняя отсутствующие нулями.
X_test_aligned = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# Также удаляем из test колонки, которые вдруг появились только в test 
X_test_final = X_test_aligned[X_train_encoded.columns]

X_train = X_train_encoded
X_test = X_test_final

print("\nРазмер X_train после кодирования:", X_train.shape)
print("Размер X_test после кодирования:", X_test.shape)
print("Доля класса 1 в train:", y_train.mean())
print("Доля класса 1 в test:", y_test.mean())


# ----------
# 2.2 Ручной расчёт энтропии, индекса Джини и прироста информации
# ----------

def entropy(labels):
    n = len(labels)
    if n == 0:
        return 0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / n
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def gini_index(labels):
    n = len(labels)
    if n == 0:
        return 0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / n
    return 1 - np.sum(probs ** 2)


def information_gain(parent_labels, left_labels, right_labels, criterion="entropy"):
    n = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)

    if criterion == "entropy":
        parent_score = entropy(parent_labels)
        left_score = entropy(left_labels)
        right_score = entropy(right_labels)
    else:
        parent_score = gini_index(parent_labels)
        left_score = gini_index(left_labels)
        right_score = gini_index(right_labels)

    weighted_child_score = (n_left / n) * left_score + (n_right / n) * right_score
    gain = parent_score - weighted_child_score

    return parent_score, weighted_child_score, gain


# --- Проверка на небольшом искусственном примере ---
print("=" * 60)
print("Проверка на искусственном примере")
print("=" * 60)

# Искусственные данные: 10 объектов, 1 признак, бинарный таргет
y_example = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 0])
feature_example = np.array([2.5, 3.1, 4.0, 1.2, 2.8, 3.5, 1.5, 4.2, 2.0, 1.8])

# Родительский узел
print("\nРодительский узел (все 10 объектов):")
print("Метки:", y_example)
print("Энтропия:", round(entropy(y_example), 4))
print("Индекс Джини:", round(gini_index(y_example), 4))

# Разбиение по порогу feature <= 2.5
threshold = 2.5
left_mask = feature_example <= threshold
right_mask = ~left_mask

y_left = y_example[left_mask]
y_right = y_example[right_mask]

print("\nРазбиение: признак <= %.1f" % threshold)
print("Левый узел (%d объектов): %s" % (len(y_left), y_left))
print("Правый узел (%d объектов): %s" % (len(y_right), y_right))

# Прирост информации по энтропии
parent_e, weighted_e, gain_e = information_gain(y_example, y_left, y_right, "entropy")
print("\nПо энтропии:")
print("  Энтропия родителя: %.4f" % parent_e)
print("  Взвешенная энтропия детей: %.4f" % weighted_e)
print("  Прирост информации: %.4f" % gain_e)

# Прирост информации по Джини
parent_g, weighted_g, gain_g = information_gain(y_example, y_left, y_right, "gini")
print("\nПо индексу Джини:")
print("  Джини родителя: %.4f" % parent_g)
print("  Взвешенный Джини детей: %.4f" % weighted_g)
print("  Прирост информации: %.4f" % gain_g)

# --- Расчёт на реальном узле обучающей выборки ---
print("\n" + "=" * 60)
print("Расчёт на реальном узле обучающей выборки")
print("=" * 60)

# Берём небольшое подмножество для наглядности (первые 100 объектов)
subset_size = 100
X_sub = X_train.iloc[:subset_size].copy()
y_sub = y_train.iloc[:subset_size].values

print("\nПодмножество: %d объектов" % subset_size)
print("Распределение target: 0=%d, 1=%d" % (
    np.sum(y_sub == 0), np.sum(y_sub == 1)))

print("\nЭнтропия родительского узла: %.4f" % entropy(y_sub))
print("Индекс Джини родительского узла: %.4f" % gini_index(y_sub))

# Выбираем числовой признак и порог
# Берём признак 'income' и порог = медиана
feature_name = "income"
if feature_name in X_sub.columns:
    feature_values = X_sub[feature_name].values
    threshold_value = np.median(feature_values)
    print("\nВыбранный признак: '%s', порог = %.2f (медиана)" % (
        feature_name, threshold_value))

    left_mask = feature_values <= threshold_value
    right_mask = ~left_mask

    y_left_real = y_sub[left_mask]
    y_right_real = y_sub[right_mask]

    print("Левый узел: %d объектов, распределение 0/1 = %d/%d" % (
        len(y_left_real), np.sum(y_left_real == 0), np.sum(y_left_real == 1)))
    print("Правый узел: %d объектов, распределение 0/1 = %d/%d" % (
        len(y_right_real), np.sum(y_right_real == 0), np.sum(y_right_real == 1)))

    parent_e, weighted_e, gain_e = information_gain(
        y_sub, y_left_real, y_right_real, "entropy")
    parent_g, weighted_g, gain_g = information_gain(
        y_sub, y_left_real, y_right_real, "gini")

    print("\nПо энтропии:")
    print("  Энтропия родителя: %.4f" % parent_e)
    print("  Взвешенная энтропия детей: %.4f" % weighted_e)
    print("  Прирост информации: %.4f" % gain_e)

    print("\nПо индексу Джини:")
    print("  Джини родителя: %.4f" % parent_g)
    print("  Взвешенный Джини детей: %.4f" % weighted_g)
    print("  Прирост информации: %.4f" % gain_g)

    print("\nВывод: прирост информации %.4f (энтропия) показывает, "
          "что разбиение по признаку '%s' с порогом %.2f "
          "уменьшает неопределённость. Чем больше прирост — "
          "тем лучше разбиение." % (gain_e, feature_name, threshold_value))
else:
    print("Признак '%s' не найден. Доступные числовые признаки:" % feature_name)
    print(X_sub.select_dtypes(include=["number"]).columns.tolist()[:10])

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# ----------
# 2.3.1 Базовая модель и сравнение критериев
# ----------

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

print("=" * 60)
print("2.3.1 Сравнение критериев gini и entropy")
print("=" * 60)

# Базовая модель с критерием gini
tree_gini = DecisionTreeClassifier(random_state=42, criterion="gini")
tree_gini.fit(X_train, y_train)

# Базовая модель с критерием entropy
tree_entropy = DecisionTreeClassifier(random_state=42, criterion="entropy")
tree_entropy.fit(X_train, y_train)

acc_gini_train = accuracy_score(y_train, tree_gini.predict(X_train))
acc_gini_test = accuracy_score(y_test, tree_gini.predict(X_test))
acc_ent_train = accuracy_score(y_train, tree_entropy.predict(X_train))
acc_ent_test = accuracy_score(y_test, tree_entropy.predict(X_test))

print("\nКритерий GINI:")
print("  Accuracy на train: %.4f" % acc_gini_train)
print("  Accuracy на test:  %.4f" % acc_gini_test)

print("\nКритерий ENTROPY:")
print("  Accuracy на train: %.4f" % acc_ent_train)
print("  Accuracy на test:  %.4f" % acc_ent_test)

# ----------
# 2.3.2 Подбор гиперпараметров max_depth и min_samples_leaf
# ----------

from sklearn.model_selection import StratifiedKFold, cross_val_score

print("\n" + "=" * 60)
print("2.3.2 Подбор гиперпараметров")
print("=" * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Подбор max_depth
max_depths = range(1, 21)
train_f1_depth = []
val_f1_depth = [] 

print("\nПодбираем max_depth")
for d in max_depths:
    print("  Проверка max_depth = %d ..." % d)
    tree = DecisionTreeClassifier(random_state=42, max_depth=d)
    
    # cross_val_score делит X_train на 5 частей, 
    # 4 учит, 1 проверяет, и так 5 раз. Возвращает массив из 5 оценок.
    val_scores = cross_val_score(tree, X_train, y_train, cv=skf, scoring='f1')
    val_f1_depth.append(val_scores.mean()) # Берем средний F1 по 5 фолдам
    
    # Для графика также считаем F1 на полном train
    tree.fit(X_train, y_train)
    train_f1_depth.append(f1_score(y_train, tree.predict(X_train), zero_division=0))

# Подбор min_samples_leaf
min_samples = [1, 5, 10, 20, 50, 100, 200, 500]
train_f1_leaf = []
val_f1_leaf = []

print("\nПодбираем min_samples_leaf")
for ms in min_samples:
    print("  Проверка min_samples_leaf = %d ..." % ms)
    tree = DecisionTreeClassifier(random_state=42, min_samples_leaf=ms)
    
    val_scores = cross_val_score(tree, X_train, y_train, cv=skf, scoring='f1')
    val_f1_leaf.append(val_scores.mean())
    
    tree.fit(X_train, y_train)
    train_f1_leaf.append(f1_score(y_train, tree.predict(X_train), zero_division=0))

# График для max_depth
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(max_depths, train_f1_depth, "b-o", label="Train")
plt.plot(max_depths, val_f1_depth, "r-o", label="Validation (CV)") # Изменили подпись
plt.xlabel("max_depth")
plt.ylabel("F1-score")
plt.title("Зависимость F1-score от max_depth")
plt.legend()
plt.grid(True)

# График для min_samples_leaf
plt.subplot(1, 2, 2)
plt.plot(min_samples, train_f1_leaf, "b-o", label="Train")
plt.plot(min_samples, val_f1_leaf, "r-o", label="Validation (CV)") # Изменили подпись
plt.xlabel("min_samples_leaf")
plt.ylabel("F1-score")
plt.title("Зависимость F1-score от min_samples_leaf")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('зависимость_от_гиперпараметров_f1.png')
print("\nГрафик сохранен в файл 'зависимость_от_гиперпараметров_f1.png'")

# Находим лучшие значения по F1-score на валидации
best_depth_idx = np.argmax(val_f1_depth)
best_leaf_idx = np.argmax(val_f1_leaf)

best_depth = max_depths[best_depth_idx]
best_leaf = min_samples[best_leaf_idx]

print("\nЛучший max_depth: %d (val F1-score = %.4f)" % (
    best_depth, val_f1_depth[best_depth_idx]))
print("Лучший min_samples_leaf: %d (val F1-score = %.4f)" % (
    best_leaf, val_f1_leaf[best_leaf_idx]))

# ----------
# 2.3.3 Обучение настроенной модели и визуализация
# ----------

print("\n" + "=" * 60)
print("2.3.3 Визуализация настроенного дерева")
print("=" * 60)

# Настраиваем итоговую модель
best_tree = DecisionTreeClassifier(
    random_state=42,
    max_depth=best_depth,
    min_samples_leaf=best_leaf
)
best_tree.fit(X_train, y_train)

print("Accuracy настроенной модели на train: %.4f" % accuracy_score(
    y_train, best_tree.predict(X_train)))
print("Accuracy настроенной модели на test: %.4f" % accuracy_score(
    y_test, best_tree.predict(X_test)))

# Визуализация дерева (ограничиваем глубину для читаемости)
plt.figure(figsize=(20, 10))
plot_tree(
    best_tree,
    max_depth=3,  # ограничиваем для читаемости
    feature_names=X_train.columns.tolist(),
    class_names=["Class 0", "Class 1"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Дерево решений (показаны первые 3 уровня)")
plt.tight_layout()
plt.savefig('Дерево_решений.png')

# ----------
# 2.3.4 Интерпретация правил для 2 объектов
# ----------

print("\n" + "=" * 60)
print("2.3.4 Интерпретация правил для 2 объектов")
print("=" * 60)

# Берём 2 объекта из тестовой выборки
sample_indices = [0, 1]
for idx in sample_indices:
    x_sample = X_test.iloc[idx:idx+1]
    y_true = y_test.iloc[idx]
    y_pred = best_tree.predict(x_sample)[0]

    print("\nОбъект #%d из тестовой выборки:" % idx)
    print("  Истинный класс: %d" % y_true)
    print("  Предсказанный класс: %d" % y_pred)

    # Получаем путь по дереву
    node_indicator = best_tree.decision_path(x_sample)
    leaf_id = best_tree.apply(x_sample)
    node_ids = node_indicator.indices[node_indicator.indptr[0]:
                                       node_indicator.indptr[1]]

    print("  Путь по дереву:")
    for node_id in node_ids:
        if leaf_id[0] == node_id:
            print("    -> ЛИСТ (узел %d)" % node_id)
        else:
            feature_idx = best_tree.tree_.feature[node_id]
            threshold_val = best_tree.tree_.threshold[node_id]
            feature_name = X_train.columns[feature_idx]
            print("    Узел %d: %s <= %.2f ?" % (
                node_id, feature_name, threshold_val))


from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

print("=" * 60)
print("2.4 Оценка качества модели")
print("=" * 60)

# Базовая модель
y_train_pred_base = tree_gini.predict(X_train)
y_test_pred_base = tree_gini.predict(X_test)

# Настраиваемая модель
y_train_pred_best = best_tree.predict(X_train)
y_test_pred_best = best_tree.predict(X_test)

def print_metrics(y_true, y_pred, label):
    """Вывод всех метрик."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("\n%s:" % label)
    print("  Accuracy:  %.4f" % acc)
    print("  Precision: %.4f" % prec)
    print("  Recall:    %.4f" % rec)
    print("  F1-score:  %.4f" % f1)
    return acc, prec, rec, f1

print("\n--- БАЗОВАЯ МОДЕЛЬ (без ограничений по глубине) ---")
print_metrics(y_train, y_train_pred_base, "Обучающая выборка")
print_metrics(y_test, y_test_pred_base, "Контрольная выборка")

print("\n--- НАСТРОЕННАЯ МОДЕЛЬ (max_depth=%d, min_samples_leaf=%d) ---" % (
    best_depth, best_leaf))
print_metrics(y_train, y_train_pred_best, "Обучающая выборка")
print_metrics(y_test, y_test_pred_best, "Контрольная выборка")

# Итоговый вывод
print("\n" + "=" * 60)
print("ИТОГОВЫЙ ВЫВОД")
print("=" * 60)

base_test_f1 = f1_score(y_test, y_test_pred_base, zero_division=0)
best_test_f1 = f1_score(y_test, y_test_pred_best, zero_division=0)
base_train_f1 = f1_score(y_train, y_train_pred_base, zero_division=0)
best_train_f1 = f1_score(y_train, y_train_pred_best, zero_division=0)

print("\n1. Базовая модель без ограничений показывает F1-score на train = %.4f "
      "и на test = %.4f." % (base_train_f1, base_test_f1))

if base_train_f1 - base_test_f1 > 0.10:
    print("   Большая разница между train и test говорит о ПЕРЕОБУЧЕНИИ.")
else:
    print("   Разница между train и test умеренная.")

print("\n2. После подбора гиперпараметров (max_depth=%d, min_samples_leaf=%d) "
      "F1-score на test = %.4f." % (best_depth, best_leaf, best_test_f1))

if best_test_f1 > base_test_f1:
    print("   Качество на контрольной выборке по F1-score УЛУЧШИЛОСЬ на %.4f." % (
        best_test_f1 - base_test_f1))
else:
    print("   Качество на контрольной выборке по F1-score изменилось на %.4f." % (
        best_test_f1 - base_test_f1))