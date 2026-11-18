# Малоранговая аппроксимация vit_base_patch16_224 на AID

Завершён исходный план: 168 ordinary SVD + 168 activation-SVD, 6 распределений
рангов, 12 дообучений с dense-контролем. Повторный подбор не проводился.
Каталог: **355 моделей**, GPU: **134 измерений** (67 × batch 1/16),
окончательный test: **20 моделей × 1500 изображений**.
Seed 42; 30 классов; frozen train/validation/test = 7000/1500/1500.
Исходные checkpoint, split и результаты сохранены без изменений.

## 1. Ранг, тип и положение матриц

Постановка: отделить эффект ранга от выбора attention/FFN и положения замены.
Эксперимент: на каждый метод 24 глобальных, 48 региональных (блоки 0–3,
4–7, 8–11) и 96 одиночных конфигураций (блоки 0/5/11), восемь рангов
.125/.25/.375/.5/.625/.75/.875/1.0. Одни и те же 1500 validation-изображений.
`all_blocks` исключает head; факторы хранятся и при увеличении числа параметров.

|model_id|total_params|validation_top1|validation_cross_entropy|
|---|---|---|---|
|exact_svd__early_ffn_r0500|78743838|0.969|0.208|
|exact_svd__late_ffn_r0500|78743838|0.969|0.137|
|exact_svd__middle_ffn_r0500|78743838|0.969|0.234|
|exact_svd__single_b0_mlp.fc1_r0125|83831070|0.950|0.291|
|exact_svd__single_b11_mlp.fc1_r0125|83831070|0.972|0.213|
|exact_svd__single_b5_mlp.fc1_r0125|83831070|0.967|0.233|
|activation_svd__early_ffn_r0500|78743838|0.971|0.199|
|activation_svd__late_ffn_r0500|78743838|0.971|0.200|
|activation_svd__middle_ffn_r0500|78743838|0.971|0.202|
|activation_svd__single_b0_mlp.fc1_r0125|83831070|0.963|0.222|
|activation_svd__single_b11_mlp.fc1_r0125|83831070|0.972|0.198|
|activation_svd__single_b5_mlp.fc1_r0125|83831070|0.971|0.199|

Глобальные результаты приведены в разделе 4. Исходная validation top-1
97.133%, CE 0.197622.
Ограничение: один checkpoint/seed/split; различия в несколько изображений
не доказывают статистически значимого преимущества.

## 2. Активации и выбор индивидуальных рангов

Постановка: использовать действие оператора на train-активациях вместо ошибки
коэффициентов. Эксперимент: 300 calibration train-изображений (10/класс),
uncentered C=E[xxᵀ], ridge 1e-4·trace(C)/in; weighted SVD и обратное преобразование;
факторы sqrt(S) симметричны в взвешенных координатах. Validation-моменты только
измеряют ошибку, не строят факторы. Калибровка и ridge взяты из завершённой серии.

Для каждого метода exact multiple-choice knapsack минимизирует сумму
ненормированных calibration local MSE по сетке рангов при точном бюджете
uniform all_blocks .25/.5/.75. Пары строк показывают изменение локальной
ошибки и validation top-1 при одном бюджете.

|rank_ratio|method|allocated|total_params|calibration_mse_sum|top1|
|---|---|---|---|---|---|
|0.250|activation_svd|False|29198622|32027.555|0.963|
|0.250|activation_svd|True|29198622|2358.218|0.942|
|0.250|exact_svd|False|29198622|124647.727|0.193|
|0.250|exact_svd|True|29198622|10691.089|0.069|
|0.500|activation_svd|False|57510174|11686.163|0.969|
|0.500|activation_svd|True|57510174|211.931|0.957|
|0.500|exact_svd|False|57510174|56651.595|0.949|
|0.500|exact_svd|True|57510174|1332.148|0.341|
|0.750|activation_svd|False|85821726|3269.637|0.971|
|0.750|activation_svd|True|85821726|13.700|0.970|
|0.750|exact_svd|False|85821726|19803.446|0.968|
|0.750|exact_svd|True|85821726|128.652|0.943|

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
|dense__dense_control__finetuned|97.133|97.267|0.198|0.204|2.000|
|exact_svd__early_ffn_r0500__finetuned|96.933|97.000|0.208|0.212|2.000|
|exact_svd__global_all_blocks_r0250__finetuned|19.333|96.000|3.293|0.244|5.000|
|exact_svd__global_all_blocks_r0500__finetuned|94.867|96.800|0.243|0.213|5.000|
|exact_svd__global_attn_r0250__finetuned|86.200|96.133|1.033|0.230|5.000|
|exact_svd__global_attn_r0500__finetuned|96.867|97.000|0.259|0.205|5.000|
|exact_svd__global_ffn_r0250__finetuned|74.733|96.333|0.938|0.224|5.000|
|exact_svd__global_ffn_r0500__finetuned|96.000|96.933|0.184|0.202|4.000|
|exact_svd__late_ffn_r0500__finetuned|96.867|97.133|0.137|0.206|4.000|
|exact_svd__middle_ffn_r0500__finetuned|96.933|97.000|0.234|0.209|1.000|
|activation_svd__global_all_blocks_r0500__finetuned|96.867|97.067|0.211|0.206|3.000|
|activation_svd__global_ffn_r0500__finetuned|97.000|96.933|0.206|0.208|2.000|

Масштаб факторов может влиять на оптимизацию. Ошибки матриц после
дообучения не измерялись; исходные локальные ошибки к ним не приписаны.

## 4. Качество, параметры, веса, память и время

Постановка: проверить реальное ускорение и стоимость хранения.
Эксперимент: вся глобальная сетка обоих методов, 6 allocated и 12 FT плюс dense;
JAX 0.4.38, FP32/highest, один процесс на model/batch,
GPU и драйвер записаны в summary.json,
batch 1/16, первые одинаковые validation-входы, 5 прогревов, 50 повторов.
Компиляция измерена отдельно; декодирование, H2D и компиляция исключены из latency.
Каждый вызов синхронизирован block_until_ready; median/P05/P95 сохранены.
Все SHA пикселей, runtime, GPU UUID и параметры протокола попарно проверены.
Прежние восемь бенчмарков повторены для единого расширенного протокола памяти.

|model_id|total_params|val_%|median_batch_ms_b1|median_batch_ms_b16|
|---|---|---|---|---|
|exact_svd__global_all_blocks_r0125|15042846|4.667|2.046|17.961|
|exact_svd__global_all_blocks_r0250|29198622|19.333|2.788|23.463|
|exact_svd__global_all_blocks_r0375|43354398|84.200|3.383|29.222|
|exact_svd__global_all_blocks_r0500|57510174|94.867|3.828|33.152|
|exact_svd__global_all_blocks_r0625|71665950|96.200|4.185|39.554|
|exact_svd__global_all_blocks_r0750|85821726|96.800|5.208|44.236|
|exact_svd__global_all_blocks_r0875|99977502|97.067|5.686|50.829|
|exact_svd__global_all_blocks_r1000|114133278|97.133|6.210|53.901|
|exact_svd__global_attn_r0125|62818590|16.133|3.766|34.791|
|exact_svd__global_attn_r0250|68127006|86.200|3.868|36.616|
|exact_svd__global_attn_r0375|73435422|95.667|4.158|38.891|
|exact_svd__global_attn_r0500|78743838|96.867|4.078|40.330|
|exact_svd__global_attn_r0625|84052254|97.000|4.570|43.036|
|exact_svd__global_attn_r0750|89360670|97.133|4.798|44.719|
|exact_svd__global_attn_r0875|94669086|97.333|5.237|47.250|
|exact_svd__global_attn_r1000|99977502|97.133|5.447|47.857|
|exact_svd__global_ffn_r0125|38045982|21.733|2.789|25.478|
|exact_svd__global_ffn_r0250|46893342|74.733|3.298|28.996|
|exact_svd__global_ffn_r0375|55740702|94.000|3.671|32.825|
|exact_svd__global_ffn_r0500|64588062|96.000|3.798|35.182|
|exact_svd__global_ffn_r0625|73435422|96.667|4.072|38.998|
|exact_svd__global_ffn_r0750|82282782|96.800|4.581|41.932|
|exact_svd__global_ffn_r0875|91130142|97.067|4.998|45.853|
|exact_svd__global_ffn_r1000|99977502|97.133|5.430|48.192|
|activation_svd__global_all_blocks_r0125|15042846|85.467|2.061|18.011|
|activation_svd__global_all_blocks_r0250|29198622|96.333|2.776|23.452|
|activation_svd__global_all_blocks_r0375|43354398|96.600|3.237|29.250|
|activation_svd__global_all_blocks_r0500|57510174|96.867|3.828|33.159|
|activation_svd__global_all_blocks_r0625|71665950|97.067|4.346|39.566|
|activation_svd__global_all_blocks_r0750|85821726|97.133|4.928|44.287|
|activation_svd__global_all_blocks_r0875|99977502|97.067|5.813|50.932|
|activation_svd__global_all_blocks_r1000|114133278|97.133|5.940|53.802|
|activation_svd__global_attn_r0125|62818590|94.733|3.781|34.802|
|activation_svd__global_attn_r0250|68127006|97.200|3.916|36.784|
|activation_svd__global_attn_r0375|73435422|96.933|4.144|38.848|
|activation_svd__global_attn_r0500|78743838|97.200|4.342|40.333|
|activation_svd__global_attn_r0625|84052254|97.267|4.554|42.964|
|activation_svd__global_attn_r0750|89360670|97.067|4.645|44.636|
|activation_svd__global_attn_r0875|94669086|97.200|4.983|47.298|
|activation_svd__global_attn_r1000|99977502|97.133|5.529|47.872|
|activation_svd__global_ffn_r0125|38045982|92.133|2.993|25.545|
|activation_svd__global_ffn_r0250|46893342|96.467|3.295|28.963|
|activation_svd__global_ffn_r0375|55740702|96.733|3.664|32.874|
|activation_svd__global_ffn_r0500|64588062|97.000|3.822|35.183|
|activation_svd__global_ffn_r0625|73435422|97.067|4.210|39.061|
|activation_svd__global_ffn_r0750|82282782|97.133|4.591|41.912|
|activation_svd__global_ffn_r0875|91130142|97.267|5.005|45.886|
|activation_svd__global_ffn_r1000|99977502|97.133|5.250|48.161|

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
На validation среди 147 неполноранговых пар accuracy activation-SVD
выше ordinary SVD в 84, равна в 28,
ниже в 35 случаях.

Список заморожен **2026-11-18T11:47:57.663131+00:00**, до чтения первого test-изображения.
SHA-256 исходного selection.json: `dc3b84a0bcd0f259aabbcc968aa0f31422742a89fc54fadf854ec253edb2fd1b`.
[Публичная запись](selected_models.json) содержит все model_id/checkpoint SHA/ранги;
исходный полный файл и manifests — в локальном 09_selection.
Включены 19 заданных моделей и дополнительный dense FT-контроль с причиной,
записанной до test. Других добавлений нет. Список после test не менялся.
Одинаковые 1500 test-изображений, их порядок, preprocessing и декодер; batch 8;
проверены SHA checkpoint/config/split/selection и попарно пиксели/метки/индексы.
Сохранены несглаженная CE, top-1 и полные локальные logits.

|model_id|val_%|test_%|test_CE|
|---|---|---|---|
|dense__dense|97.133|97.667|0.182|
|exact_svd__global_all_blocks_r0250|19.333|19.733|3.224|
|exact_svd__global_all_blocks_r0375|84.200|84.133|0.663|
|exact_svd__global_all_blocks_r0500|94.867|95.000|0.224|
|exact_svd__global_all_blocks_r0750|96.800|97.267|0.180|
|exact_svd__global_attn_r0500|96.867|97.133|0.249|
|exact_svd__global_ffn_r0500|96.000|96.667|0.161|
|dense__dense_control__finetuned|97.267|96.800|0.216|
|exact_svd__global_all_blocks_r0500__finetuned|96.800|97.133|0.198|
|exact_svd__global_ffn_r0500__finetuned|96.933|97.067|0.194|
|activation_svd__allocated_activation_svd_all_blocks_r0500|95.733|95.867|0.251|
|exact_svd__allocated_exact_svd_all_blocks_r0500|34.067|33.733|2.430|
|activation_svd__global_all_blocks_r0250|96.333|96.067|0.232|
|activation_svd__global_all_blocks_r0375|96.600|97.333|0.200|
|activation_svd__global_all_blocks_r0500|96.867|97.867|0.192|
|activation_svd__global_all_blocks_r0750|97.133|97.733|0.185|
|activation_svd__global_attn_r0500|97.200|97.667|0.185|
|activation_svd__global_ffn_r0500|97.000|97.800|0.190|
|activation_svd__global_all_blocks_r0500__finetuned|97.067|97.067|0.196|
|activation_svd__global_ffn_r0500__finetuned|96.933|96.867|0.197|

Ограничения: 1500 наблюдений и один исходный checkpoint. Test использован один раз
для фиксированного списка; по его результатам не выбирались конфигурации,
эпохи, ранги, ridge или новые эксперименты. После test выполнялась только
агрегация и проверка артефактов. Дополнительные GPU-эксперименты не запускались.

Машиночитаемые данные: [все модели](models.csv), [GPU](benchmarks.csv), [test](test.csv),
[пары](paired_comparison.csv), [бюджеты](budget_comparison.csv), [FT](finetuning.csv).
Команды: [FINAL_STAGES.md](../../FINAL_STAGES.md).
Полная локальная доказательная база: `artifacts/experiments/aid_base_final_20261118/`;
исходные серии остаются на прежних путях. Презентация не изменялась.
