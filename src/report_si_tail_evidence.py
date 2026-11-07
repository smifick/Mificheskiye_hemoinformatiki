"""Build the evidence report and a standalone scientific figure from executed outputs."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_si_tail_evidence import strata


def number(value):
    return f'{value:,.2f}'.replace(',',' ')


def markdown(frame):
    columns=list(frame.columns)
    rows=['| '+' | '.join(map(str,columns))+' |','| '+' | '.join(['---']*len(columns))+' |']
    for values in frame.itertuples(index=False,name=None):
        rows.append('| '+' | '.join(number(v) if isinstance(v,(float,np.floating)) else str(v) for v in values)+' |')
    return '\n'.join(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',default='results/si_tail_evidence')
    parser.add_argument('--report',default='docs/si_tail_evidence.md')
    args=parser.parse_args()
    root=Path(args.results)
    audit=json.loads((root/'audit/summary.json').read_text())
    identity=json.loads((root/'audit/identity_noise.json').read_text())
    external=json.loads((root/'external_audit/summary.json').read_text())
    final=json.loads((root/'final/selection_and_validation.json').read_text())
    verification=json.loads((root/'verification.json').read_text())
    metrics=pd.read_csv(root/'benchmark/metrics_by_stratum.csv')
    controls=pd.read_csv(root/'final/comparison_with_nonnegative_controls.csv')
    component=pd.read_csv(root/'benchmark/component_predictions.csv')
    oof=pd.read_csv(root/'benchmark/oof_predictions.csv')
    intervals=json.loads((root/'final/uncertainty.json').read_text())
    external_scores=pd.read_csv(root/'external_ablation/comparison.csv')
    old=[]
    for seed,path in [(2718,'legacy_audit'),(7349,'legacy_audit_repeat')]:
        table=pd.read_csv(root/path/'comparison.csv');table['seed']=seed;old.append(table)
    old=pd.concat(old,ignore_index=True)
    paired=metrics[metrics.stratum.eq('all')].pivot(index='model',columns='seed',values='MSE')
    paired['mean_two_seeds']=paired.mean(axis=1)
    paired=paired.reset_index().sort_values('mean_two_seeds')
    paired.columns=[str(c) for c in paired.columns]
    summary=controls[controls.model.isin(['baseline_krr_nonnegative','baseline_lgbm_nonnegative','smearing_nonnegative','mean_glm_nonnegative'])]
    summary=summary.groupby(['model','stratum'],as_index=False).MSE.mean().pivot(index='model',columns='stratum',values='MSE')
    summary=summary[['all','physical_low','physical_medium','physical_tail','technical_10000']].reset_index()
    primary=oof[(oof.seed==2718)&(oof.row_index==14)][['model','observed','prediction','squared_error']]
    peak=component[(component.seed==2718)&(component.row_index==14)&component.model.eq('smearing')].iloc[0]
    pub=pd.read_csv(root/'external_ablation/publication_holdout.csv')
    pub_rows=[]
    for flag,frame in pub.groupby('with_old_external'):
        pub_rows.append({'old_external':bool(flag),'n':len(frame),
                         'physical_MSE':float(np.mean((frame.physical_prediction-frame.observed_SI)**2)),
                         'organizer_train_n':int(frame.organizer_train_rows.iloc[0]),
                         'external_train_n':int(frame.external_train_rows.iloc[0])})
    ci=next(row for row in intervals if row['model']=='smearing_nonnegative' and row['reference']=='baseline_krr_nonnegative')
    baseline=summary.set_index('model').loc['baseline_krr_nonnegative']
    new=summary.set_index('model').loc['smearing_nonnegative']
    low_change=100*(new.physical_low/baseline.physical_low-1)
    medium_change=100*(new.physical_medium/baseline.physical_medium-1)
    budget=[]
    for seed in sorted(oof.seed.unique()):
        base_name=final['strongest_baseline'].removesuffix('_nonnegative')
        base_frame=oof[(oof.seed==seed)&oof.model.eq(base_name)].sort_values('row_index')
        new_frame=oof[(oof.seed==seed)&oof.model.eq('smearing')].sort_values('row_index')
        y=new_frame.observed.to_numpy()
        delta=(np.maximum(new_frame.prediction.to_numpy(),0)-y)**2-(np.maximum(base_frame.prediction.to_numpy(),0)-y)**2
        for name,mask in strata(y,new_frame.label_kind.to_numpy(),new_frame.tail_threshold_from_training.to_numpy()).items():
            budget.append({'seed':int(seed),'stratum':name,'n':int(mask.sum()),
                           'MSE_contribution_removed':float(-delta[mask&(delta<0)].sum()/len(y)),
                           'MSE_contribution_added':float(delta[mask&(delta>0)].sum()/len(y)),
                           'net_MSE_contribution_change':float(delta[mask].sum()/len(y))})
    budget=pd.DataFrame(budget)
    budget.to_csv(root/'final/paired_error_budget.csv',index=False)
    tail_table=metrics[(metrics.seed==2718)&metrics.stratum.eq('physical_tail')][['model','n','MSE','bias']]
    text=f'''# SI VACV: причины занижения, новые модели и внешние данные

Дата: 07.11.2026. Все новые численные оценки ниже вычислены кодом и сохранены в `{root}`. Исторические Public-оценки из README не являются новыми результатами этого исследования.

**Устойчивое улучшение физического хвоста с сохранением остального качества не подтверждено.** У новых моделей есть отдельные снижения общей MSE, но они не дают согласованного выигрыша по физическому хвосту, малым/средним SI и неопределённости. Финальный строгий структурный артефакт сохраняет KRR, а новый кандидат и его CSV сохранены отдельно. Историческая v7 с признаком присутствия в ChEMBL также проверена на тех же фолдах; она не заменялась новым более слабым решением.

## 1. Данные и действительные ограничения задачи

Использованы README, история репозитория и скриншот Dataset Description пользователя. Описание предусматривает сбор открытых молекулярных данных и требует SMILES/IC50/CC50/SI/DOI. Оно запрещает организаторские `CC50_uM` и `IC50_uM_VACV` как предикторы. Ранее проверенные Kaggle URL возвращали 404; после предоставления скриншота это не трактуется как запрет внешнего дополнения.

VACV означает вирус вакцинии. Данные других вирусов из тех же публикаций не добавлялись. Работа ограничена опубликованными измерениями, качеством данных и статистическими моделями; лабораторные инструкции не извлекались.

Из 320 организаторских строк **237** имеют обе концентрации и согласованный SI, **81** содержит технический код 10000 при отсутствующем IC50, **2** — технический ноль при отсутствующем CC50. Все 320 меток сохраняются в основной MSE. Концентрации и метаданные используются только в отдельном аудите/отборе внешних данных; fit новых моделей получает структуры и разрешённую цель.

Подлинная крупнейшая метка — **36066.667**, row_index=14, source_row=16; она равна `1082/0.03` при единицах мкМ и контексте Vero. Значение 36666.667 в постановке — опечатка. Исходный DOI и операторы сравнения организаторских концентраций отсутствуют. Сходная статья упоминается в литературе, но конкретная пара структура–измерение не подтверждена. Метка не удаляется и не объявляется ошибкой.

После нормализации 320 различных строк SMILES дают 312 структур. Есть {identity['materially_conflicting_identity_groups']} группы с существенными противоречиями, включая одну структуру с SI=39.984 и 10000. Минимальная эмпирическая MSE для детерминированного прогноза по этой нормализованной структуре составляет {number(identity['empirical_minimum_MSE_for_deterministic_canonical_structure_predictor'])}. Это граница для данных/представления, а не оценка популяционного Bayes risk. Найдены {identity['morgan_bit_patterns_shared_by_distinct_structures']} Morgan-паттерна, общие разным каноническим структурам; экстремальная строка в эти коллизии не входит.

## 2. Воспроизведение прежних результатов

На тех же 5 scaffold-фолдах seed=2718:

{markdown(pd.DataFrame([{'model':name,**value} for name,value in audit['reproduction'].items()]))}

Расхождение OOF-прогнозов с сохранёнными v5-файлами — порядка 1e-12. Переход от прежних baseline к новым методам не смешивается с изменением разбиений.

## 3. Почему хвост не восстанавливается автоматически

В проблемном фолде максимум числовой обучающей цели — 10000, физического SI — 5000. Ближайшие разрешённые соседи экстремальной структуры имеют Tanimoto 0.527 и 0.491, но SI 68.130 и 134. Ни сходство, ни соседние метки не подтверждают величину 36067.

В исторической v6/v7 обнаружена верхняя проекция до 10000. Однако её снятие не изменило ни один OOF-прогноз: до границы они не доходили. Следовательно, это потенциальное ограничение реализации, но не подтверждённая причина текущих прогнозов около 300. Новые модели верхней границы не имеют. Отдельный синтетический пример LightGBM даёт {audit['boosting_counterexample']['prediction']:.6f} при max(y_train)=1: любые boosting-модели не объявляются универсально ограниченными обучающим максимумом.

Логарифмический прогноз имеет другую проблему: обратное преобразование среднего логарифма не равно условному среднему исходного Y. Исправление этого смещения проверено, но не создаёт отсутствующие сведения о зависимости больших физических SI от структуры.

Разложение новой смеси для известного экстремума особенно показательно: общий прогноз {peak.prediction:.2f}, вероятность технического кода {peak.p_code:.6f}, вклад `10000*p_code` около {10000*peak.p_code:.2f}, физический компонент **{peak.physical_SI:.2f}**. Оценённый фактор обратного преобразования равен {peak.smearing_factor:.6f}. Численное приближение к большому SI в основном связано с кодовым режимом; физический хвост модель не выучила.

## 4. Источники и три направления проверки

Полная [библиография с выводами, допущениями и ограничениями](si_tail_evidence_sources.md) отделяет полные тексты, аннотации и ранее курированные данные.

1. **Пределы реализации:** сравнить исторический capped и uncapped прогноз при одинаковом fit, исключив внешние записи проверочных структур/остовов и внешние test-идентичности. Никакого lookup SI проверочных молекул. Изменение результата оказалось нулевым.
2. **Непосредственная модель условного среднего:** регуляризованный log-link GLM по signed-log RDKit-дескрипторам; Poisson/Tweedie/Gamma, power=1/1.5/2, alpha=1/10/100. По Santos Silva–Tenreyro и документации GLM это модель среднего в исходной шкале, а не log-OLS. Сдвиг Y+1 обратим и поддерживает два числовых нуля. Масштабирование, импутация, масштаб цели и выбор параметров — внутри обучения.
3. **Структурно-зависимое обратное преобразование:** известный технический режим и физический компонент; centered KRR по Morgan+torsion, gate с Morgan/feature-Morgan, внутренние scaffold-OOF остатки. Контроли — без поправки и глобальная Duan-style поправка. Новая поправка — сглаженное условное среднее exp(остатка) с K²-сходствами и prior_count=5/25; alpha физического KRR=3/10. Величина поправки оценивается по обучающим остаткам, произвольный множитель экстремуму не задаётся.

Это не попытка обучить редкий физический tail-gate: одна наблюдаемая метка выше 10000 не позволяет оценить его вероятность и условную величину. GPD/EVT не подгонялась по одному экстремуму. Масса 10000 не интерпретируется как набор физических превышений.

## 5. Основная оценка и выбор

5 внешних scaffold-фолдов, 3 внутренних для выбора кандидатов, для поправки обратного преобразования — отдельный grouped cross-fit. Повторы seed=2718 и заранее выбранный 7349 используют те же 320 молекул и не являются независимыми выборками. 117 generic Bemis–Murcko групп; одинаковые канонические таутомеры объединяются. Сохраняются внешние, внутренние и калибровочные разбиения. Все оценки и выбор параметров — невзвешенная raw MSE. 19 фиксированных конфигураций, без ранней остановки по внешним меткам.

{markdown(paired)}

`smearing` и `mean_glm` оценивают процедуру внутреннего выбора семейства, `nested_selector` дополнительно выбирает между семействами. Это оценка конкретных процедур; последующее ранжирование по внешним OOF не превращает победителя в независимо подтверждённую модель.

Сравнение средних по двум повторам, с общей для всех моделей проекцией отрицательных прогнозов в ноль:

{markdown(summary)}

При неотрицательной цели проекция отрицательных прогнозов в ноль не может увеличить квадрат ошибки. Это фиксированная граница области цели, не обрезание больших прогнозов. Raw-оценки сохранены отдельно. Относительно сильнейшего прежнего структурного baseline KRR новая смесь ухудшает MSE малых SI на **{low_change:.2f}%**, средних — на **{medium_change:.2f}%** в среднем по повторам. Поэтому снижение общей MSE само по себе не удовлетворяет поставленному ограничению качества.

Парный bootstrap по scaffold-группам, с усреднением потерь повторов одного объекта до ресэмплинга: разность MSE смеси и KRR = {number(ci['mean_MSE_difference'])}, 95% интервал [{number(ci['cluster_bootstrap_95pct'][0])}; {number(ci['cluster_bootstrap_95pct'][1])}]. Интервал включает ноль. Это условная диагностика сохранённых OOF-ошибок без переобучения во bootstrap; история выбора представлений и гипотез не устраняется таким интервалом.

Бюджет изменения ошибки смеси относительно KRR на seed=2718, в единицах вклада в общую MSE (деление на 320):

{markdown(budget[(budget.seed==2718)&~budget.stratum.isin(['all','physical_all'])].drop(columns='seed'))}

Так видно отдельно, сколько ошибки устранено и сколько добавлено в каждой непересекающейся части цели. Полный бюджет обоих повторов сохранён в `final/paired_error_budget.csv`.

## 6. Известный экстремум и независимость вывода

Прогнозы при истинном SI=36066.667 на основном разбиении:

{markdown(primary)}

Этот пример уже изучался раньше и не является независимым подтверждением. Его вклад и крупнейшие ошибки остальных строк сохранены в `largest_error_contributions.csv`. Ошибки кодов, физического хвоста и остальных SI разделены. Диагностический хвост — физические значения выше 95-го процентиля обучающей части; технические коды из расчёта порога исключены. Число таких проверочных строк различается между повторами (17 и 12), поэтому определения и n сохраняются рядом с метриками.

MSE и смещение физического хвоста на основном разбиении:

{markdown(tail_table)}

## 7. Дополнительная проверка экстраполяции

В каждом внешнем обучающем фолде вычислен 90-й процентиль физического SI и удалены целые обучающие scaffold-группы, содержащие значения выше него. Валидационные структуры не меняются. Это диагностическая смена распределения обучения, а не замена основной CV и не основание удалить экстремум из финального обучения. Новая подгонка гиперпараметров выполняется только на оставшемся обучении.

Сохранены отдельные метрики выше фактического физического диапазона обучения, максимумы/квантили, число физических хвостовых строк и наибольший прогноз. Техническая числовая масса 10000 может остаться в обучении, но не называется физическим верхним диапазоном. Результаты этой проверки не использованы при финальном выборе.

## 8. Внешние данные: что действительно добавилось

Прежние 56 курированных строк — 38 структур, включая 38 Vero-строк и 18 повторных A549-контекстов. Они не считаются заново найденными. Проверены совпадения с организаторскими train/test, прежней курированной базой и всеми {external['previous_raw_unique_canonical_structures']} каноническими структурами прежней сырой ChEMBL-выгрузки.

Новый DOI `{external['new_source_doi']}`: {external['all_table_rows']} строк полной VACV Table 1; {external['exact_rows']} точных пар, {external['excluded_bounds_or_inactive']} исключения NA/границ. **{external['new_exact_structures_not_in_previous_raw_endpoints']} новых молекул и {external['new_scaffolds_vs_train_and_all_previous_endpoints']} новый остов** относительно train и прежней выгрузки. Два новых SI выше 10000, максимум {external['new_SI_max']:.6f}. Сохраняются исходные концентрации, единицы, операторы, SD, опубликованный SI, рассчитанное отношение средних, DOI и степень атрибуции структуры. Для этого DOI не выдумываются ChEMBL activity ID.

Доля SI>10000 составляет 2/14 в новой точной таблице против 1/237 среди организаторских измеренных отношений. Источник был найден целевым поиском хвоста; полная таблица снижает отбор отдельных удачных соединений, но не устраняет публикационное смещение и корреляцию аналогов. При знаменателях 0.03±0.01 и 0.04±0.01 мкМ большой SI чувствителен к точности знаменателя. Это не основания заменить точные средние границами, удалить их или объявить ошибками.

После исключения test-идентичностей исследовательский банк содержит {external['bank_research_rows_after_test_exclusion']} строк. На каждом внешнем и внутреннем фолде дополнительно исключаются совпадающие валидационные структуры и остовы. Проверены веса источников 0, 0.1, 1; code-gate не обучается на внешних физических SI. Внешние концентрации и метаданные не являются входными признаками.

{markdown(external_scores[external_scores.stratum.eq('all')][['model','n','MSE','RMSE','MAE']])}

Улучшение от новой публикации при весе 0.1 очень мало и не меняет итоговый выбор. Старые или объединённые внешние записи не дают преимущества контрольной конфигурации на основной MSE. Сохранена отдельная ретроспективная проверка всей новой публикации: исключены все её структуры/остовы из организаторского обучения и прежнего внешнего банка. Проверяется только физический компонент; её MSE не является соревновательной оценкой:

{markdown(pd.DataFrame(pub_rows))}

## 9. Историческая сильная модель v6/v7

Дополнительно прочитана и воспроизведена ветка `codex/v7-regression-20261107-333336`, а не только README текущей main. На тех же разбиениях, без lookup валидационных SI:

{markdown(old[['seed','model','MSE','extreme_prediction']])}

Эта модель использует признак присутствия структуры в ChEMBL и старый банк уровней V/C; tier C не проходит новую независимую сверку публикационной пары. Вариант `no_external_labels` всё ещё использует source-presence, что прямо отмечено в его протоколе. Эти результаты — более сильные исторические сравнительные точки, но не доказательство физического переноса по одной структуре: часть качества связана с происхождением технической кодировки. Ни один новый метод не объявляется превосходящим эту модель. Она и её прежние submission-файлы не заменялись; в строгом новом протоколе метаданные оставлены для отбора/аудита, а не как source-membership признак.

## 10. Сохранённый итог и ограничения

Строгий структурный контрольный артефакт: **{final['recommended_model']}**, средняя MSE двух повторов **{number(final['mean_seed_MSE'])}**. В финальном fit сохранены все 320 организаторских строк. Внешние метки испытаны и не включены автоматически при отсутствии подтверждённого преимущества. Новая смесь/условная поправка и её отдельный submission сохранены в `benchmark/` для проверки, без заявления улучшения Kaggle.

`final/submission.csv`: {final['submission_rows']} строк, правильные уникальные индексы и порядок sample; диапазон [{final['submission_min']:.6f}, {final['submission_max']:.6f}]; верхней границы нет. Перезагрузка модели воспроизводит прогнозы. Ни значения sample-файла, ни test-концентрации, ни test-метки не используются для калибровки. Public/Private score нового файла неизвестен; автоматической отправки в Kaggle не было.

47 тестов и 8 существующих subtests прошли. Независимо пересчитаны {verification['metric_rows_independently_recomputed']} строк метрик; пересечений внешних, внутренних и калибровочных групп нет. CSV обоих артефактов воспроизведены из сохранённых моделей, хеши записаны в `verification.json`. Финальный структурный CSV отличается от прежнего `submission_candidate_v3.csv` не более чем на {final.get('max_abs_difference_vs_existing_v3_submission',float('nan')):.3g}; это воспроизведение baseline, не новый заявленный выигрыш.

Не хватает подтверждённого происхождения организаторского экстремума, сопоставимых опубликованных высоких физических SI на нескольких переносимых структурных группах и ясного разделения кодировки/физических измерений. Дополнительные независимые публикационные данные нужны для проверки этой экстраполяции; произвольный коэффициент для одной известной молекулы не заменяет такую информацию.

![Сравнение MSE и физических OOF-прогнозов](../{root}/tail_diagnostics.png)

На правой диаграмме применяется только фиксированная проекция отрицательных прогнозов в ноль; верхняя часть диапазона не обрезается.

## Воспроизведение

Из корня репозитория, Python 3.12; выходные каталоги должны быть новыми:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt -r requirements-si-tail-evidence.txt
HF_HOME=work/hf_cache HF_HUB_DISABLE_XET=1 python src/audit_si_tail_evidence.py --out results/evidence_reproduced/audit
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python src/benchmark_si_tail_evidence.py --out results/evidence_reproduced/benchmark
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python src/benchmark_si_tail_evidence.py --out results/evidence_reproduced/extrapolation --diagnostic --no-final
python src/curate_si_tail_evidence.py --out results/evidence_reproduced/external_audit
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python src/ablate_external_si_tail.py --bank results/evidence_reproduced/external_audit/research_training_bank.csv --out results/evidence_reproduced/external_ablation
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python src/analyze_si_tail_evidence.py --benchmark results/evidence_reproduced/benchmark --out results/evidence_reproduced/final
python src/predict_si_tail_evidence.py --model results/evidence_reproduced/final/model.joblib --input data/raw/test.csv --out work/repeated_submission.csv
python -m pytest tests -q
python src/validate_si_tail_evidence.py --results results/evidence_reproduced
python src/report_si_tail_evidence.py --results results/evidence_reproduced --report work/reproduced_evidence_report.md
```

Для исторической проверки:

```bash
git fetch origin codex/v7-regression-20261107-333336
git worktree add --detach work/reference_v7 FETCH_HEAD
python src/audit_legacy_si_tail.py --legacy-root work/reference_v7 --out results/evidence_reproduced/legacy_audit
python src/audit_legacy_si_tail.py --legacy-root work/reference_v7 --seed 7349 --out results/evidence_reproduced/legacy_audit_repeat
```

Фактический commit и хеши исходников исторической модели записаны в её protocol.json. Основной новый benchmark не зависит от этой ветки. Скрипт первоначальной ChEMBL-выгрузки, её DOI/activity ID и журнал запросов уже находятся в `src/download_external.py` и `data/external/`; новый сборщик добавляет отдельную полную публикационную таблицу.
'''
    Path(args.report).write_text(text)
    os.environ.setdefault('MPLCONFIGDIR',str(Path('work/matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['baseline_krr','baseline_lgbm','mean_glm','smearing']
    labels=['KRR','LightGBM','Mean GLM','Regime + smearing']
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.6),layout='constrained')
    positions=np.arange(len(names))
    for j,seed in enumerate([2718,7349]):
        values=metrics[(metrics.stratum=='all')&(metrics.seed==seed)].set_index('model').loc[names].MSE/1e6
        axes[0].barh(positions+(j-.5)*.3,values,height=.28,label=f'Scaffold seed {seed}')
    axes[0].set(yticks=positions,yticklabels=labels,xlabel='Unweighted MSE (million)',title='Paired scaffold evaluation')
    axes[0].invert_yaxis();axes[0].legend(fontsize=8);axes[0].set_xlim(left=0)
    for name,color in [('baseline_krr','#376897'),('smearing','#bf6a35')]:
        data=oof[(oof.seed==2718)&oof.model.eq(name)&oof.label_kind.eq('measured_ratio')]
        axes[1].scatter(data.observed,np.maximum(data.prediction,0),s=15,alpha=.55,label=name,color=color)
    axes[1].plot([0,40000],[0,40000],color='gray',lw=1,ls='--')
    axes[1].set_xscale('symlog',linthresh=1);axes[1].set_yscale('symlog',linthresh=1)
    axes[1].set(xlabel='Observed physical SI',ylabel='OOF prediction',title='Measured ratios only; seed 2718')
    axes[1].legend(fontsize=8)
    fig.savefig(root/'tail_diagnostics.png',dpi=180)
    plt.close(fig)
    print(args.report)


if __name__=='__main__':
    main()
