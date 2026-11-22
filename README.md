# Digital heart twin: GNN-эмулятор биомеханики сердца

Материалы к ВКР:
графовая нейронная сеть HeartSimSage [15] как суррогатная модель
конечно-элементного анализа (КЭА) пассивной механики бивентрикулярного
миокарда.

## Структура

| Путь | Назначение |
|---|---|
| `heartsim_sage.py` | Учебная реализация архитектуры из отчёта (разделы 2.3–3.1): граф из сетки, поля Лапласа–Дирихле, стратифицированные окрестности, 5 модулей сети, обучение на подмножествах, персонализация β. Демонстрация — на реальной сетке. |
| `generate_illustrations.py` | Рисунки к разделам 2–3 (реальная сетка). |
| `src/prepare_meshes.py` | Подготовка сеток из открытого набора Rodero et al. (2021), Zenodo 4590294. |
| `src/fix_face_orientation.py` | Исправление ориентации граней в готовых `data/geom/*.npz`. |
| `src/fem_core.py`, `src/fem_solver.py` | Нелинейный КЭ-решатель пассивного наполнения (давление на эндокарде, условия Робена на эпикарде и базе). |
| `src/verify_solver.py` | Проверки решателя (конечные разности, patch-тест). |
| `src/generate_dataset.py` | Набор решений КЭА с контролем роста объёма полостей. |
| `src/graph_features.py`, `src/hss_model.py`, `src/train_hss.py` | Признаки графа, модель и обучение эмулятора (разбиение по пациентам). |
| `src/fig_mesh.py`, `src/fig_results.py`, `src/make_report.py` | Рис. 2.1, рисунки и сводка результатов эксперимента. |
| `data/geom`, `data/fea` | 11 сеток и 176 решений КЭА. |
| `results/` | Метрики, предсказания на тесте, веса модели, `ОТЧЁТ.md`. |

## Воспроизведение

```bash
pip install numpy scipy torch matplotlib pypardiso   # pypardiso необязателен
python src/verify_solver.py
python src/fix_face_orientation.py                    # только для старых geom-файлов
cd src
python generate_dataset.py --n-per-geom 16            # ~45 мин на 2 ядрах
python train_hss.py --epochs 80
python fig_mesh.py && python fig_results.py && python make_report.py
cd .. && python generate_illustrations.py && python heartsim_sage.py
```

Подготовка сеток с нуля (`src/prepare_meshes.py`) дополнительно требует
`pyvista`, `pyacvd`, `tetgen` и загрузки исходных данных с Zenodo.

## Источники данных

- Rodero C. et al. Virtual cohort of adult healthy four-chamber heart meshes
  from CT images. Zenodo, 2021, record 4590294 (CC BY 4.0).
- Shi L., Chen Y., Vedula V. HeartSimSage. arXiv:2504.18968, 2025 — [15].

Что изменено по замечаниям руководителя — см. `CHANGELOG.md`.
