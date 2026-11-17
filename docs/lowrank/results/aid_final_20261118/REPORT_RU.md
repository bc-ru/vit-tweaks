# Итог: низкоранговые ViT-Small на AID

Завершён исходный план: 168 ordinary SVD + 168 activation-SVD, 6 распределений
рангов, 12 дообучений с dense-контролем. Повторный подбор не проводился.
Каталог: **355 моделей**, GPU: **134 измерения** (67 × batch 1/16),
окончательный test: **20 моделей × 1500 изображений**.
Seed 42; 30 классов; frozen train/validation/test = 7000/1500/1500.
Исходные checkpoint, split и результаты сохранены без изменений.

## 1. Ранг, тип и положение матриц

Постановка: отделить эффект ранга от выбора attention/FFN и положения замены.
Эксперимент: на каждый метод 24 глобальных, 48 региональных (блоки 0–3,
4–7, 8–11) и 96 одиночных конфигураций (блоки 0/5/11), восемь рангов
.125/.25/.375/.5/.625/.75/.875/1.0. Одни и те же 1500 validation-изображений.
`all_blocks` исключает head; факторы хранятся и при увеличении числа параметров.

Результат: ordinary all_blocks .5 даёт 65.000%, activation — 96.067%;
ordinary FFN .5 — 85.800%, attention .5 — 95.133%. При равном числе параметров
ordinary regional FFN .5: early 94.600%, middle 96.533%, late 97.000%.
Для одиночного FFN fc1 .125: блок 0 — 79.667%, блок 11 — 97.333%.
Положение существенно, единый локальный ранг не задаёт качество сети.
Все 42 полносетевых полноранговых контроля двух методов прошли допуски
relative logits 5e-5 / max abs 1e-3.
Ограничение: один checkpoint/seed/split; различия в несколько изображений
не доказывают статистически значимого преимущества.

## 2. Активации и выбор индивидуальных рангов

Постановка: использовать действие оператора на train-активациях вместо ошибки
коэффициентов. Эксперимент: 300 calibration train-изображений (10/класс),
uncentered C=E[xxᵀ], ridge 1e-4·trace(C)/in; weighted SVD и обратное преобразование;
факторы sqrt(S) симметричны в взвешенных координатах. Validation-моменты только
измеряют ошибку, не строят факторы. Калибровка и ridge взяты из завершённой серии.

Результат: calibration и независимая validation local MSE лучше в 336/336
неполноранговых парах; Frobenius весов — в 0/336. Для каждого метода exact
multiple-choice knapsack распределял ранги по той же сетке при точном бюджете
uniform all_blocks .25/.5/.75, минимизируя сумму ненормированных train local MSE.
**Во всех шести случаях меньшая сумма локальных ошибок ухудшила accuracy сети.**
На бюджете .5: ordinary 65.000% → 12.600%, activation 96.067% → 89.800%;
локальные суммы соответственно 49343 → 3006.3 и 11544 → 869.01.
Это отрицательный результат выбранного суррогатного критерия.
Ограничение: после совместных замен активации меняются; локальные ошибки
не складываются в ошибку композиции. Решение DP оптимально только для его
дискретной локальной задачи, не для accuracy сети.

## 3. Восстановление дообучением

Постановка: измерить фактическое восстановление, отделяя его от дополнительных
эпох dense. Эксперимент: прежние 9 ordinary + 2 activation + dense; все параметры,
фиксированные ранги, 5 эпох, batch 8, FP32, AdamW (.9/.999, eps 1e-8), LR 1e-5
с cosine по эпохам, WD .05, train smoothing .1, seed 42. CE оценки несглаженная.
Лучшая эпоха 1–5 по validation top-1, ранняя при равенстве. Нового обучения нет.

|model_id|before_%|after_%|ce_before|validation_cross_entropy|best_epoch|
|---|---|---|---|---|---|
|dense__dense_control__finetuned|97.333|97.200|0.198|0.195|3.000|
|exact_svd__early_ffn_r0500__finetuned|94.600|97.067|0.308|0.207|5.000|
|exact_svd__global_all_blocks_r0250__finetuned|5.600|85.800|3.499|0.559|5.000|
|exact_svd__global_all_blocks_r0500__finetuned|65.000|95.800|1.356|0.233|5.000|
|exact_svd__global_attn_r0250__finetuned|39.667|95.200|2.202|0.264|4.000|
|exact_svd__global_attn_r0500__finetuned|95.133|96.600|0.277|0.211|4.000|
|exact_svd__global_ffn_r0250__finetuned|18.733|92.867|3.029|0.322|4.000|
|exact_svd__global_ffn_r0500__finetuned|85.800|96.867|0.604|0.215|5.000|
|exact_svd__late_ffn_r0500__finetuned|97.000|97.533|0.220|0.188|3.000|
|exact_svd__middle_ffn_r0500__finetuned|96.533|96.800|0.207|0.219|2.000|
|activation_svd__global_all_blocks_r0500__finetuned|96.067|96.667|0.243|0.217|3.000|
|activation_svd__global_ffn_r0500__finetuned|96.467|97.133|0.231|0.202|5.000|

Ограничение: восстановление неполно для сильного сжатия. У middle_ffn accuracy
растёт, но CE ухудшается; dense-контроль accuracy немного теряет. Масштаб факторов
activation отличается и может влиять на оптимизацию. Ошибки матриц после
дообучения не измерялись; исходные локальные ошибки к ним не приписаны.

## 4. Качество, параметры, веса, память и время

Постановка: проверить реальное ускорение и стоимость хранения.
Эксперимент: вся глобальная сетка обоих методов, 6 allocated и 12 FT плюс dense;
RTX 4070, JAX 0.4.38, driver 555.42.02, FP32/highest, один процесс на model/batch,
batch 1/16, первые одинаковые validation-входы, 5 прогревов, 50 повторов.
Компиляция измерена отдельно; декодирование, H2D и компиляция исключены из latency.
Каждый вызов синхронизирован block_until_ready; median/P05/P95 сохранены.
Все SHA пикселей, runtime, GPU UUID и параметры протокола попарно проверены.
Прежние восемь бенчмарков повторены для единого расширенного протокола памяти.

|Модель|FP32 веса, MB|NPZ, MB|Batch 1, ms|Batch 16, ms|XLA план b16, MB|
|---|---:|---:|---:|---:|---:|
|dense__dense|86.709|86.757|1.790|13.329|145.048|
|exact_svd__global_all_blocks_r0500|58.397|58.485|1.820|11.627|116.743|
|activation_svd__global_all_blocks_r0500|58.397|58.486|1.798|11.726|116.743|
|exact_svd__global_all_blocks_r0500__finetuned|58.397|58.487|1.813|11.610|116.743|
|activation_svd__global_all_blocks_r0500__finetuned|58.397|58.488|1.822|11.624|116.743|
|exact_svd__allocated_exact_svd_all_blocks_r0500|58.397|58.510|1.856|11.392|116.743|
|activation_svd__allocated_activation_svd_all_blocks_r0500|58.397|58.510|1.846|11.357|116.743|

У all_blocks .5 параметров на 32.65% меньше исходной модели. Измеренное отношение median dense/model для batch 16: 1.146× (SVD) и 1.137× (activation). Для batch 1 ускорения у этих двух исходных конфигураций нет. MB = 10⁶ байт; столбец XLA — статический план, не runtime-пик.

|model_id|total_params|val_%|median_batch_ms_b1|median_batch_ms_b16|
|---|---|---|---|---|
|exact_svd__global_all_blocks_r0125|3982494|1.933|1.310|7.733|
|exact_svd__global_all_blocks_r0250|7521438|5.600|1.432|8.764|
|exact_svd__global_all_blocks_r0375|11060382|23.267|1.671|10.471|
|exact_svd__global_all_blocks_r0500|14599326|65.000|1.820|11.627|
|exact_svd__global_all_blocks_r0625|18138270|92.933|1.983|12.948|
|exact_svd__global_all_blocks_r0750|21677214|96.133|2.100|14.380|
|exact_svd__global_all_blocks_r0875|25216158|97.133|2.219|15.759|
|exact_svd__global_all_blocks_r1000|28755102|97.333|2.350|16.457|
|exact_svd__global_attn_r0125|15926430|3.133|1.686|11.696|
|exact_svd__global_attn_r0250|17253534|39.667|1.745|12.176|
|exact_svd__global_attn_r0375|18580638|86.467|1.812|12.657|
|exact_svd__global_attn_r0500|19907742|95.133|1.937|13.115|
|exact_svd__global_attn_r0625|21234846|96.467|1.971|13.628|
|exact_svd__global_attn_r0750|22561950|97.000|2.005|14.113|
|exact_svd__global_attn_r0875|23889054|97.067|2.017|14.680|
|exact_svd__global_attn_r1000|25216158|97.333|2.037|14.803|
|exact_svd__global_ffn_r0125|9733278|7.467|1.452|9.168|
|exact_svd__global_ffn_r0250|11945118|18.733|1.524|10.029|
|exact_svd__global_ffn_r0375|14156958|48.000|1.651|11.303|
|exact_svd__global_ffn_r0500|16368798|85.800|1.691|11.970|
|exact_svd__global_ffn_r0625|18580638|95.200|1.778|12.726|
|exact_svd__global_ffn_r0750|20792478|96.600|1.950|13.777|
|exact_svd__global_ffn_r0875|23004318|97.133|1.987|14.551|
|exact_svd__global_ffn_r1000|25216158|97.333|2.084|14.971|
|activation_svd__global_all_blocks_r0125|3982494|24.667|1.303|7.423|
|activation_svd__global_all_blocks_r0250|7521438|85.133|1.462|8.821|
|activation_svd__global_all_blocks_r0375|11060382|94.533|1.660|10.545|
|activation_svd__global_all_blocks_r0500|14599326|96.067|1.798|11.726|
|activation_svd__global_all_blocks_r0625|18138270|96.600|1.960|12.852|
|activation_svd__global_all_blocks_r0750|21677214|96.733|2.160|14.637|
|activation_svd__global_all_blocks_r0875|25216158|96.867|2.201|15.859|
|activation_svd__global_all_blocks_r1000|28755102|97.333|2.359|16.620|
|activation_svd__global_attn_r0125|15926430|87.533|1.681|11.604|
|activation_svd__global_attn_r0250|17253534|95.867|1.753|11.936|
|activation_svd__global_attn_r0375|18580638|96.533|1.807|12.639|
|activation_svd__global_attn_r0500|19907742|96.867|1.933|13.285|
|activation_svd__global_attn_r0625|21234846|96.933|1.974|13.706|
|activation_svd__global_attn_r0750|22561950|97.067|1.986|14.162|
|activation_svd__global_attn_r0875|23889054|97.067|1.999|14.745|
|activation_svd__global_attn_r1000|25216158|97.333|2.034|15.052|
|activation_svd__global_ffn_r0125|9733278|34.600|1.467|9.448|
|activation_svd__global_ffn_r0250|11945118|89.867|1.523|9.898|
|activation_svd__global_ffn_r0375|14156958|95.000|1.645|11.133|
|activation_svd__global_ffn_r0500|16368798|96.467|1.709|11.906|
|activation_svd__global_ffn_r0625|18580638|96.733|1.788|12.805|
|activation_svd__global_ffn_r0750|20792478|97.000|1.903|13.778|
|activation_svd__global_ffn_r0875|23004318|96.933|2.019|14.629|
|activation_svd__global_ffn_r1000|25216158|97.333|2.047|15.064|

`models.csv` объединяет все 355 моделей, точные ранги/имена (ranks_json),
validation CE/top-1, ошибки логитов, идентификаторы checkpoint, результаты FT,
latency и память. `benchmarks.csv` содержит все 134 строки, включая allocated/FT.
`matrix_errors.csv` и `local_errors.csv` — исходные измеренные матричные ошибки.
Аналитические Frobenius/spectral-кривые ordinary SVD сохранены отдельно в
`svd_matrix_curves.csv`, исходные ошибки замен — в `svd_matrix_errors.csv`.

Различаем четыре величины: `param_memory_bytes` — точные FP32 байты параметров;
`checkpoint_bytes` — размер NPZ с метаданными; `process_pool_peak_bytes` —
максимальный резерв GPU-аллокатора процесса; `process_allocator_peak_bytes` —
его максимум занятых байтов за загрузку/компиляцию/прогрев/инференс.
`compiled_plan_bytes` — статическая сумма XLA arguments+output+temp−alias
([документация JAX](https://docs.jax.dev/en/latest/notebooks/host-offloading.html)),
а не измеренный runtime-пик. **Изолированный фактический peak GPU инференса
достоверно не измерен: inference_gpu_peak_bytes = null.** Эти величины не заменяют
друг друга и не включают одинаковые накладные расходы драйвера.
Уменьшение параметров не называется ускорением без соответствующего latency.
Частоты GPU не фиксировались; это одна серия на одной карте, интервалы P05/P95 —
разброс повторов, не доверительные интервалы между независимыми сериями.
Региональные/одиночные исходные варианты не бенчмаркировались отдельно;
измерены их уже дообученные варианты. Размер NPZ не равен размеру массива весов.

Графики validation: [параметры](accuracy_parameters.png), [байты весов](accuracy_weights.png),
[latency batch 1](accuracy_latency_b1.png), [latency batch 16](accuracy_latency_b16.png).

## 5. Ordinary SVD против activation-SVD и окончательный test

Постановка: сравнить одинаковые архитектуру, ранги, входы и стоимость хранения;
проверить заранее определённые модели на held-out test.
На validation среди 147 неполноранговых пар activation лучше в 91, равен в 25,
хуже в 31 случае. Локальное улучшение не гарантирует улучшения полной модели.
На all_blocks .5 после FT: 95.800% ordinary / 96.667% activation;
на FFN .5: 96.867% / 97.133%. Это результаты validation, не критерий по test.

Список заморожен **2026-11-17T20:12:29.315244+00:00**, до чтения первого test-изображения.
SHA-256 исходного selection.json: `e6a1e7aa80256fc30785eb68e2ec03fa74ae0ce571dc65e48301171ef50cec01`.
[Публичная запись](selected_models.json) содержит все model_id/checkpoint SHA/ранги;
исходный полный файл и manifests — в локальном 09_selection.
Включены 19 заданных моделей и дополнительный dense FT-контроль с причиной,
записанной до test. Других добавлений нет. Список после test не менялся.
Одинаковые 1500 test-изображений, их порядок, preprocessing и декодер; batch 8;
проверены SHA checkpoint/config/split/selection и попарно пиксели/метки/индексы.
Сохранены несглаженная CE, top-1 и полные локальные logits.

|model_id|val_%|test_%|test_CE|
|---|---|---|---|
|dense__dense|97.333|96.800|0.203|
|exact_svd__global_all_blocks_r0250|5.600|4.667|3.497|
|exact_svd__global_all_blocks_r0375|23.267|22.067|2.957|
|exact_svd__global_all_blocks_r0500|65.000|63.267|1.402|
|exact_svd__global_all_blocks_r0750|96.133|95.933|0.248|
|exact_svd__global_attn_r0500|95.133|94.267|0.292|
|exact_svd__global_ffn_r0500|85.800|84.867|0.624|
|dense__dense_control__finetuned|97.200|96.600|0.206|
|exact_svd__global_all_blocks_r0500__finetuned|95.800|95.600|0.245|
|exact_svd__global_ffn_r0500__finetuned|96.867|96.067|0.235|
|activation_svd__allocated_activation_svd_all_blocks_r0500|89.800|88.733|0.451|
|exact_svd__allocated_exact_svd_all_blocks_r0500|12.600|12.133|3.242|
|activation_svd__global_all_blocks_r0250|85.133|82.733|0.704|
|activation_svd__global_all_blocks_r0375|94.533|93.400|0.313|
|activation_svd__global_all_blocks_r0500|96.067|96.133|0.235|
|activation_svd__global_all_blocks_r0750|96.733|96.800|0.215|
|activation_svd__global_attn_r0500|96.867|97.000|0.202|
|activation_svd__global_ffn_r0500|96.467|96.000|0.235|
|activation_svd__global_all_blocks_r0500__finetuned|96.667|96.133|0.226|
|activation_svd__global_ffn_r0500__finetuned|97.133|96.400|0.220|

Ограничения: 1500 наблюдений и один исходный checkpoint. Test использован один раз
для фиксированного списка; по его результатам не выбирались конфигурации,
эпохи, ранги, ridge или новые эксперименты. После test выполнялась только
агрегация и проверка артефактов. Дополнительные GPU-эксперименты не запускались.

Машиночитаемые данные: [все модели](models.csv), [GPU](benchmarks.csv), [test](test.csv),
[пары](paired_comparison.csv), [бюджеты](budget_comparison.csv), [FT](finetuning.csv).
Команды: [FINAL_STAGES.md](../../FINAL_STAGES.md).
Полная локальная доказательная база: `artifacts/experiments/aid_final_20261118/`;
исходные серии остаются на прежних путях. Презентация не изменялась.
