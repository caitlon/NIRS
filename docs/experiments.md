# Запуск экспериментов в NIRS

Этот документ описывает как запускать эксперименты в проекте NIRS, используя новый интерфейс экспериментов.

## Обзор системы экспериментов

Система экспериментов в NIRS построена вокруг концепции конфигурационных файлов, которые определяют все аспекты эксперимента: от загрузки данных до обучения модели и сохранения результатов. Такой подход обеспечивает:

- **Воспроизводимость**: каждый эксперимент полностью определен его конфигурацией
- **Прослеживаемость**: конфигурация сохраняется вместе с результатами
- **Гибкость**: легко создавать и изменять эксперименты без изменения кода
- **Масштабируемость**: запускать множество экспериментов пакетно

## Способы запуска экспериментов

### 1. Запуск через командную строку

Самый простой способ запустить эксперимент - использовать скрипт `run_experiment.py`:

```bash
# Запустить один эксперимент
python experiments/run_experiment.py --config configs/pls_snv_savgol.yaml

# Запустить все эксперименты в директории
python experiments/run_experiment.py --config_dir configs/

# Запустить с подробным выводом
python experiments/run_experiment.py --config configs/rf_msc_feature_selection.yaml --verbose
```

### 2. Запуск из Python-кода

Вы также можете запускать эксперименты программно из вашего Python-кода:

```python
# Запуск одного эксперимента
from experiments.experiment_manager import run_experiment

results = run_experiment("configs/pls_snv_savgol.yaml")
print(f"R2 score: {results['metrics']['r2_score']:.4f}")

# Запуск нескольких экспериментов
from experiments.experiment_manager import run_experiments

all_results = run_experiments("configs/")
for name, result in all_results.items():
    if "error" in result:
        print(f"Error in {name}: {result['error']}")
    else:
        print(f"{name}: R2 score = {result['metrics']['r2_score']:.4f}")
```

### 3. Программное создание и запуск экспериментов

Для более сложных сценариев вы можете создавать конфигурации программно:

```python
from experiments.experiment_manager import ExperimentManager
from nirs_tomato.config import ExperimentConfig, DataConfig, ModelConfig, FeatureSelectionConfig

# Создание конфигурации
config = ExperimentConfig(
    name="custom_pls_experiment",
    description="Custom PLS regression experiment",
    data=DataConfig(
        data_path="data/raw/my_dataset.csv",
        target_column="Brix",
        transform="snv",
    ),
    model=ModelConfig(
        model_type="pls",
        pls_n_components=12,
        test_size=0.25,
    ),
    feature_selection=FeatureSelectionConfig(
        method="vip",
        n_features=30,
        plot_selection=True
    )
)

# Запуск эксперимента
manager = ExperimentManager()
results = manager.run_from_config_object(config)

# Анализ результатов
print(f"R2 score: {results['metrics']['r2_score']}")
print(f"RMSE: {results['metrics']['rmse']}")
```

## Создание конфигураций экспериментов

### Структура YAML-файла конфигурации

Конфигурационные файлы используют формат YAML и имеют следующую структуру:

```yaml
name: experiment_name
description: Experiment description

# Конфигурация данных
data:
  data_path: data/raw/dataset.csv
  target_column: Brix
  transform: snv  # snv, msc, или none
  savgol:
    enabled: true
    window_length: 15
    polyorder: 2
    deriv: 1
  remove_outliers: false
  exclude_columns:
    - Notes
    - Timestamp
    - Lab

# Конфигурация выбора признаков
feature_selection:
  method: vip  # none, ga, cars, или vip
  n_features: 20
  plot_selection: true
  vip_n_components: 10

# Конфигурация модели
model:
  model_type: pls  # pls, svr, rf, xgb, или lgbm
  tune_hyperparams: false
  test_size: 0.2
  random_state: 42
  
  # Параметры для PLS
  pls_n_components: 10
  
  # Параметры для SVR
  svr_kernel: rbf
  svr_C: 1.0
  
  # Параметры для RF
  rf_n_estimators: 100
  rf_max_depth: 10

# Конфигурация вывода
output_dir: models
results_dir: results
verbose: false

# Конфигурация MLflow
mlflow:
  enabled: true
  experiment_name: nirs-tomato-experiments
```

### Создание новой конфигурации

1. Скопируйте существующий файл конфигурации:

```bash
cp configs/pls_snv_savgol.yaml configs/my_new_experiment.yaml
```

2. Отредактируйте параметры в соответствии с вашими потребностями
3. Запустите эксперимент с новой конфигурацией

### Валидация конфигураций

Все конфигурации автоматически валидируются с использованием Pydantic, что гарантирует корректность параметров.

## Результаты экспериментов

После запуска эксперимента результаты сохраняются в директории, указанной в параметре `results_dir` (по умолчанию `results/`):

- `{experiment_name}_predictions.csv` - предсказания модели
- `{experiment_name}_metrics.txt` - метрики производительности
- `{experiment_name}_config.yaml` - использованная конфигурация
- `{experiment_name}_selected_features.csv` - выбранные признаки (если применимо)

Обученная модель сохраняется в директории, указанной в параметре `output_dir` (по умолчанию `models/`):
- `{experiment_name}.pkl` - сериализованная модель

## Отслеживание экспериментов с MLflow

Если параметр `mlflow.enabled` установлен в `true`, информация об эксперименте будет логироваться в MLflow:

1. Запустите MLflow UI:

```bash
python experiments/run_mlflow_server.py
```

2. Откройте http://127.0.0.1:5000 в браузере для просмотра результатов экспериментов

MLflow сохраняет:
- Все параметры конфигурации
- Метрики (R2, RMSE, MAE и др.)
- Графики и артефакты
- Обученные модели

## Примеры конфигураций

В директории `configs/` находятся примеры конфигураций для различных сценариев:

- `pls_snv_savgol.yaml` - PLS-регрессия с SNV-преобразованием и фильтром Савицкого-Голея
- `rf_msc_feature_selection.yaml` - Random Forest с MSC-преобразованием и VIP-отбором признаков
- `xgb_genetic_algorithm.yaml` - XGBoost с генетическим алгоритмом отбора признаков

## Интеграция с собственным кодом

Если вы хотите интегрировать систему экспериментов с вашим кодом, используйте `ExperimentManager`:

```python
from experiments.experiment_manager import ExperimentManager
from nirs_tomato.config import ExperimentConfig

# Создайте менеджер экспериментов
manager = ExperimentManager()

# Загрузите конфигурацию
config = ExperimentConfig.from_yaml("configs/my_config.yaml")

# Модифицируйте конфигурацию программно
config.model.tune_hyperparams = True
config.model.test_size = 0.3

# Запустите эксперимент
results = manager.run_from_config_object(config)

# Используйте результаты
model = results["model"]
new_predictions = model.predict(new_data)
```

## Советы и рекомендации

- **Именование конфигураций**: Используйте осмысленные имена, включающие ключевые аспекты эксперимента
- **Версионирование конфигураций**: Храните конфигурации в системе контроля версий
- **Организация экспериментов**: Группируйте эксперименты по цели или датасету
- **Документирование**: Добавляйте детальное описание в поле `description`
- **Воспроизводимость**: Всегда устанавливайте `random_state` для воспроизводимых результатов
- **Итеративная разработка**: Начинайте с простых конфигураций и постепенно усложняйте 