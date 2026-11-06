"""Generate the Russian explanation document from recorded experiment artifacts."""
import argparse,json
from pathlib import Path
from docx import Document
from docx.shared import Pt

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args();root=Path(a.root)
    run=root/'results/final_run';m=json.loads((run/'metrics.json').read_text());audit=json.loads((run/'data_audit.json').read_text())
    ext=json.loads((root/'data/external/collection_summary.json').read_text());validation=json.loads((root/'results/validation_report.json').read_text())
    doc=Document();doc.add_heading('VACV: обучение регрессии индекса селективности',0)
    doc.add_paragraph('6 ноября 2026 года. Репозиторий: https://github.com/smifick/Structure-based-de-novo-design-')
    doc.add_heading('Что получено',1)
    doc.add_paragraph(f'Модель обучена на всех {audit["train_rows"]} строках организаторского train после отдельной оценки качества. Подготовлен submission.csv для {audit["test_rows"]} строк test. Метрика Kaggle — MSE. По предоставленному пользователем скриншоту первая отправка получила Public MSE 9 992 163.26222. Автоматическая отправка из этого кода не выполнялась; private-результат неизвестен.')
    doc.add_heading('Выполненные действия — построчно',1)
    steps=[
      'Прочитаны train.csv (320 строк), test.csv (597 строк), sample_submit.csv (597 строк). Проверены схема, цель и порядок идентификаторов.',
      'Выбрана цель SI VACV и проверены конечность и неотрицательность меток. Признаки получаются только из SMILES; CC50, IC50, название, культура и штамм не используются.',
      'Проверены все SMILES: ошибочных строк нет. Для 13 многокомпонентных обучающих записей выбрана наиболее крупная углеродсодержащая часть; выполнена нейтрализация зарядов RDKit Uncharger. Исходный SMILES сохранён.',
      'После стандартизации в train осталось 312 разных родительских структур. Строки повторов сохранены из-за разных экспериментальных условий, но всегда относятся к одной scaffold-группе.',
      'Выявлено 81 значение SI=10000, каждое совпадает с пропуском IC50. Пользователь подтвердил: это служебный код отсутствующего IC50. Метки сохранены для воспроизведения задания Kaggle; это не доказанные значения селективности.',
      'Для 237 записей доступны IC50 и CC50; отношение совпадает с SI с относительным допуском 2%. Пропущенные исходные концентрации не восстановлены вымышленными значениями.',
      'Построены Morgan fingerprints: радиус 2, 2048 бит, с учётом хиральности. Для части сравниваемых моделей рассчитаны дескрипторы RDKit. Пропуски дескрипторов заполняются только внутри обучающего фолда.',
      'Построены 117 обобщённых каркасов Bemis–Murcko. Ациклические соединения объединены в одну группу. Разбиение удерживает связанные структуры в одном фолде; оно не гарантирует полного отсутствия структурного сходства между группами.',
      f'Отделена отложенная выборка: {m["holdout_rows"]} строк; оставшиеся {m["outer_train_rows"]} строк использованы во внутренней GroupKFold-валидации из пяти фолдов. Seed=42.',
      'Сравнены 16 конфигураций: константные прогнозы, Random Forest, Extra Trees, Histogram Gradient Boosting, Ridge и Kernel Ridge с Tanimoto-ядром, включая исходную и log1p-шкалу цели.',
      f'По объединённым out-of-fold предсказаниям внутренней CV выбран {m["selected_model"]}. Выбор выполнен по MSE без использования holdout. Гиперпараметр alpha=1.',
      'Выбранная модель оценена на holdout один раз. После оценки она переобучена на всех 320 строках для финального submission; оценочная и финальная модели имеют разный объём обучающих данных.',
      'Предсказания обрезаны снизу нулём; после этого проверены конечность, две колонки index и SI VACV, 597 строк и порядок индексов из шаблона.',
      'Сохранён model.joblib и написан отдельный predict.py. Повторный прогноз из сохранённой модели совпал с submission с допуском 1e-12.',
      f'Выполнена выгрузка ChEMBL: {ext["vacv_assays"]} анализ; {ext["all_unique_endpoint_records"]} уникальных endpoint-записей, все с DOI. Записаны URL и SHA256 ответов API.',
      'CC50 дополнительно запрашивался по публикациям с VACV-активностью. Такие CC50 не являются отдельными противовирусными измерениями; контекст сохранён в таблице.',
      'Концентрации переведены в мкМ только из распознаваемых молярных единиц. Значения с > или < сохранены с оператором, но не считаются точными метками. EC50 не переименовывается в IC50.',
      f'Сформированы {ext["candidate_pairs"]} возможных пар концентраций в одной публикации для одного ChEMBL-соединения: 152 пары с IC50 и 185 с EC50. Каждая имеет рассчитанное отношение и статус проверки контекста.',
      'Для всех найденных пар контекст/время эксперимента не подтверждены достаточно полно. Отношения сохранены как предварительные кандидаты, без автоматического допуска в обучение. Два прямых точных SI из ChEMBL равны 0 и также требуют проверки первоисточника.',
      'Финальный обучающий набор состоит из предоставленных 320 строк; внешние метки не добавлялись. Файл training_dataset.csv содержит SMILES, IC50_uM, CC50_uM, SI, doi, исходные поля и флаги. DOI организаторских записей отсутствуют и оставлены пустыми.',
      'Проверены независимые инварианты: отсутствие пересечения scaffold-групп, схема submission, единицы/отношения внешних записей, формат DOI, точность вычисления отношений.',
      'Запущены пять модульных тестов: нормализация солей, группировка стереоизомеров, ошибочный SMILES, свойства Tanimoto-ядра, проверка числовых значений и времени.',
      'Подготовлены README, исходный код, внешняя база, фактический обучающий набор, модель, метрики и пояснительный документ для размещения в GitHub.'
    ]
    for i,step in enumerate(steps,1):doc.add_paragraph(f'{i:02d}. {step}')
    doc.add_heading('Метрики',1)
    table=doc.add_table(rows=1,cols=5);table.style='Light Shading Accent 1'
    for cell,val in zip(table.rows[0].cells,['Проверка','MSE','RMSE','MAE','R²']):cell.text=val
    for label,vals in [('Внутренняя CV, выбранная модель',m['inner_cv']),('Holdout, выбранная модель',m['holdout']),('Holdout, константа train mean',m['holdout_dummy_mean'])]:
        cells=table.add_row().cells
        for cell,val in zip(cells,[label]+[f'{vals[k]:,.3f}' for k in ['MSE','RMSE','MAE','R2']]):cell.text=val
    reduction=100*(1-m['holdout']['MSE']/m['holdout_dummy_mean']['MSE'])
    doc.add_paragraph(f'MSE на holdout ниже константного прогноза на {reduction:.1f}%. Это одна локальная отложенная выборка, а не результат Kaggle. Селекция среди 16 вариантов делает оценку внутренней CV оптимистичнее независимой оценки; размер holdout мал.')
    doc.add_heading('Внешние данные и ограничения',1)
    doc.add_paragraph('Данные: https://www.ebi.ac.uk/chembl/api/data/ . Запросы приведены в request_manifest.json; DOI и ссылки на записи — в CSV. Условия использования: '+ext['license_url'])
    doc.add_paragraph('Внешняя база содержит неполные и цензурированные измерения. Пустое поле означает отсутствие подтверждённого значения. external_si_database.csv содержит 152 предварительных отношения CC50/IC50, а не 152 проверенные обучающие метки. Автоматическое объединение таких данных с организаторскими метками SI=10000 может ухудшить сопоставимость цели.')
    doc.add_paragraph('После стандартизации 30 родительских структур присутствуют и в train, и в test. Это свойство предоставленного разбиения; test не участвовал в выборе модели. Для локальной валидации повторы и стереоизомеры удерживаются в одной scaffold-группе.')
    doc.add_paragraph('Индекс селективности не доказывает отсутствие действия на другие объекты организма. Прогнозы не подтверждают эффективность или безопасность препарата. Оценка неопределённости модели не калибрована: max_train_tanimoto в диагностике — только показатель структурного сходства.')
    diagnostic=root/'results/error_analysis/summary.json'
    if diagnostic.exists():
        doc.add_heading('Разбор ошибок после первой отправки Kaggle',1)
        doc.add_paragraph('Public MSE первой отправки: 9 992 163.26222. По скриншоту результат лидера: 1 546 537.62825. Его метод неизвестен; причины разрыва нельзя определить только по лидерборду.')
        doc.add_paragraph('На локальном holdout 13 кодов отсутствующего IC50 дают 55.8% суммы квадратов ошибок. При метке 10000 средний прогноз равен 4918.26. Для остальных 39 строк средний SI равен 127.06, средний прогноз — 2088.39. Пример ошибки: SI=5.111, прогноз=6666.17.')
        doc.add_paragraph('Код 10000 не описывает величину или границу биологического SI. Отсутствие IC50 не означает неактивности. При обучении на таких метках модель частично решает задачу предсказания кодировки пропуска. Два SI=0 совпадают с отсутствующим CC50 и требуют отдельного уточнения.')
        doc.add_paragraph('Сохранён отдельный поднабор 237 строк с обеими концентрациями, для дальнейшего научного анализа. Текущая модель и submission не менялись. Следующий эксперимент — разделить вероятность служебной метки и условный числовой SI; улучшение этим подходом ещё не проверено. Изученный holdout больше нельзя использовать как нетронутую оценку при дальнейшем подборе.')
    doc.add_heading('Как воспроизвести',1)
    commands=['python -m pip install -r requirements.txt','python src/download_external.py --out data/external --cache work/chembl_cache','python src/train_model.py --train data/raw/train.csv --test data/raw/test.csv --sample data/raw/sample_submit.csv --out results/reproduced_run','python src/validate_data.py --run results/final_run --sample data/raw/sample_submit.csv --external data/external --out results/validation_report.json','python src/predict.py --model results/final_run/model.joblib --input data/raw/test.csv --out results/repeated_submission.csv','python -m unittest discover -s tests -v']
    for command in commands:
        par=doc.add_paragraph();run=par.add_run(command);run.font.name='Consolas';run.font.size=Pt(9)
    doc.add_paragraph('Каталог --out при обучении должен быть новым. Для повторной проверки нового запуска укажите его каталог в --run. Документ пересоздаётся командой python src/make_report.py --root .')
    doc.add_heading('Проверки',1);doc.add_paragraph(json.dumps(validation,ensure_ascii=False,indent=2))
    (root/'docs').mkdir(exist_ok=True);doc.save(root/'docs/explanation_ru.docx')
    (root/'docs/explanation_ru.txt').write_text('\n'.join(f'{i:02d}. {step}' for i,step in enumerate(steps,1)),encoding='utf-8')

if __name__=='__main__':main()
