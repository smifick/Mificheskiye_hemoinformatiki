# Воспроизведение эксперимента

## Окружение

Python 3.12. Версии зависимостей соответствуют сохранённому эксперименту и закреплены в `requirements.txt`. Расчёты выполняются на CPU. Сохранённый `joblib` предназначен для доверенной загрузки в совместимом окружении.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Для Windows активация окружения: `.venv\Scripts\activate`.

## Выгрузка ChEMBL

```bash
python src/download_external.py --out data/external --cache work/chembl_cache
```

Требуется доступ к ChEMBL REST API. Кэш содержит исходные JSON-ответы и используется при повторном запросе. В `data/external/request_manifest.json` фиксируются URL и SHA256. При обновлении ChEMBL состав новой выгрузки может измениться; сохранённые CSV относятся к зафиксированному набору результатов.

## Обучение и локальная оценка

```bash
python src/train_model.py --train data/raw/train.csv --test data/raw/test.csv --sample data/raw/sample_submit.csv --out results/reproduced_run
```

Каталог `--out` должен отсутствовать до запуска; существующие эксперименты не перезаписываются. Результат включает параметры, разбиения, внутренние OOF-прогнозы, отложенные прогнозы, модель и submission. Внешние метки текущим обучающим скриптом не используются.

## Проверка артефактов

```bash
python src/validate_data.py --run results/final_run --sample data/raw/sample_submit.csv --external data/external --out results/validation_report.json
python -m unittest discover -s tests -v
```

Для нового эксперимента в `--run` указывается его каталог. Операторы цензурирования, формат DOI и арифметические отношения проверяются отдельно от сопоставимости экспериментальных условий.

## Прогноз из сохранённой модели

```bash
python src/predict.py --model results/final_run/model.joblib --input data/raw/test.csv --out results/repeated_submission.csv
```

Обязательная входная колонка: `SMILES`. При наличии `index` он переносится в выходной файл; иначе создаётся последовательность индексов. Остальные входные поля в признаки не включаются.

## Диагностика ошибок и отчёт

```bash
python src/analyze_errors.py --root .
python src/make_report.py --root .
```

Диагностика использует сохранённые OOF- и отложенные прогнозы и не выполняет подбор модели. DOCX формируется из `docs/technical_report.md`; содержание отчёта относится к эксперименту `results/final_run`.

## Дополнительные эксперименты v2

Конфигурации, команды и ограничения интерпретации приведены в [experiment_v2.md](experiment_v2.md). Для inference кандидата: `python src/predict_v2.py --model results/v2/model_v2.joblib --input data/raw/test.csv --out results/repeated_v2.csv`.
