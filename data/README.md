# Данные

## Исходные таблицы

| Файл | Строк | Назначение |
|---|---:|---|
| [raw/train.csv](raw/train.csv) | 320 | Организаторские структуры, концентрации и метки SI VACV |
| [raw/test.csv](raw/test.csv) | 597 | Организаторские тестовые структуры и сопутствующие поля |
| [raw/sample_submit.csv](raw/sample_submit.csv) | 597 | Схема и порядок индексов submission |

DOI и первичные публикации для организаторских строк не предоставлены. Числа 10000 при отсутствующем IC50 представляют технический код. Два нуля SI совпадают с отсутствующим CC50. Эти строки не являются подтверждёнными отношениями концентраций.

## Производные таблицы

[processed/training_dataset.csv](processed/training_dataset.csv) — точный состав 320 строк текущего обучающего эксперимента. Поля включают исходный `SMILES`, `canonical_smiles`, `scaffold_group`, `source_row`, `IC50_uM`, `CC50_uM`, `SI`, `doi`, `SI_possible_placeholder` и исходные колонки. DOI отсутствуют; имя `SI_possible_placeholder` сохранено для совместимости с первоначальной схемой.

[processed/si_with_observed_concentrations.csv](processed/si_with_observed_concentrations.csv) — 237 строк с обеими концентрациями и дополнительным `SI_from_concentrations`. Поднабор не проходил независимую проверку первичных публикаций и пока не использовался для отдельного обучения.

## Внешняя выгрузка ChEMBL

| Файл | Содержание |
|---|---|
| [external/collected_database.csv](external/collected_database.csv) | 6510 измерений: SMILES, IC50_uM, CC50_uM, SI, doi, EC50_uM, endpoint, relation, validation_status, source_url |
| [external/external_endpoints.csv](external/external_endpoints.csv) | Полная таблица измерений с assay/document/molecule ID, клеточной линией и описанием анализа |
| [external/external_si_database.csv](external/external_si_database.csv) | 152 предварительных отношения CC50/IC50; все требуют проверки условий |
| [external/candidate_si_pairs.csv](external/candidate_si_pairs.csv) | 337 предварительных пар с отдельным указанием IC50 или EC50 в знаменателе |
| [external/direct_si.csv](external/direct_si.csv) | Два прямых точных SI; оба нулевые и требуют проверки первоисточника |
| [external/vacv_assays.json](external/vacv_assays.json) | Метаданные 621 VACV-анализа |
| [external/request_manifest.json](external/request_manifest.json) | 44 URL-запроса и SHA256 ответов |
| [external/collection_summary.json](external/collection_summary.json) | Агрегированная статистика и политика отбора |

В длинном формате одна строка соответствует одному endpoint. Пустые IC50/CC50/SI означают отсутствие такого измерения в данной записи. Значения с `relation` > или < являются границами. EC50 хранится отдельно от IC50. Вычисленное отношение предварительной пары не считается допущенной обучающей меткой.

Источник: https://www.ebi.ac.uk/chembl/api/data/ . DOI и URL записи приведены построчно. Условия использования: https://chembl.gitbook.io/chembl-interface-documentation/about/terms-of-use .
