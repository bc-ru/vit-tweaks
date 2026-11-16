# Реализованные этапы 00–02 (JAX)

Три независимые команды; общего запуска экспериментов нет. Используется точная
float32 SVD и симметричное распределение sqrt(S) между факторами. Старые результаты,
checkpoint и split не изменяются. CPU выбран по умолчанию. GPU допускается только
явно (`--device gpu`) после проверки отсутствия вычислительных процессов через
`nvidia-smi`; перед GPU-запуском также проверьте доступность карты вручную.

## Среда и реальные входы

Команды ниже выполняются из корня репозитория. Они используют существующий
`aerial-svd-jax:cpu`; зависимости на хост устанавливать не требуется.
`--user` сохраняет доступность файлов на хосте. В CPU-образе нет Git: SHA передаётся
через `AERIAL_SVD_GIT_COMMIT`. Если Git доступен, команда читает revision сама.
Хэши всех исходных Python-файлов сохраняются дополнительно: один commit не описывает
незакоммиченные изменения. При внешней передаче SHA состояние working tree отмечено
как неизвестное, а не как clean.

Обнаруженный обученный native checkpoint: `.scratch/jax-migration/model-vit_small.npz`.
Для доступного пользователю стабильного пути создана побитовая копия
`checkpoints/jax/vit_small_protocol_v1.npz` (исходник был root-owned с mode 0600).
Старый файл и его права сохранены; SHA-256 копии совпал. Первичная полная
проверка 00 использовала исходный scratch-путь, команды ниже используют копию.
Это **не** `.scratch/jax-migration/vit_small.npz`: последний является архивом трёх
матриц, а не моделью. Для долговременного хранения можно повторить конвертацию
исходного `runs/aid_vit_small/best.pt` существующей командой `convert_checkpoint`
(см. `JAX_MIGRATION.md`), обязательно в новый путь, и явно передать его вместо scratch.
Автоматической конвертации/подмены checkpoint нет.

Общие параметры: `--config`, `--checkpoint`, `--output-dir` обязательны;
`--device {cpu,gpu}` необязателен. Пути разрешаются относительно рабочего каталога.
Каждому повторному запуску нужен **новый** каталог. Приведённые пути уже заняты
результатами проверки; для повторения измените идентификатор запуска.

## 00_baseline

```bash
docker run --rm --user "$(id -u):$(id -g)" --network none \
  -e AERIAL_SVD_GIT_COMMIT="$(git rev-parse HEAD)" \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=2 \
  -v "$PWD:/workspace" -w /workspace aerial-svd-jax:cpu \
  python -m aerial_svd.lowrank_baseline \
  --config configs/aid_vit_small.yaml \
  --checkpoint checkpoints/jax/vit_small_protocol_v1.npz \
  --output-dir experiments/lowrank_2026/protocol_v1_cpu_20261117/00_baseline \
  --device cpu --batch-size 8 \
  --expected-top1 0.9733333333333334 --top1-atol 0.005
```

Выходы: `baseline.json`, `manifest.json`. Проверяются архитектура, seed, плотная
структура, классы, полное непересекающееся разбиение, fingerprint (если записан в
checkpoint), хэш исходного Torch checkpoint (если записан конвертером), конечность
метрик и повторяемость логитов первой validation-порции. Оценка только на `val`,
без аугментаций, float32, несглаженная cross-entropy. Ожидаемый top-1 и допуск
задаются до оценки; `reference_verified` означает именно успешное сравнение с ним.
Допуск 0.005 допускает отличие до 7 изображений из 1500 при переходе с Torch/AMP;
в выполненной проверке top-1 совпал точно.

Для короткого smoke можно убрать `--expected-top1` и добавить
`--max-val-batches 1 --batch-size 1`. Частичная оценка маркируется
`quality_verification=partial_smoke_only`; её нельзя выдавать за полную базовую
оценку. `--expected-top1` с ограничением числа порций запрещён.

Split уже содержит train/val/test (7000/1500/1500), 30 классов. Подразбиение не
создавалось. Для другого исходника с train/test команда остановится: перед
использованием нужен отдельно сохранённый validation из train, без изменения test.

## 01_factor_checks

```bash
docker run --rm --user "$(id -u):$(id -g)" --network none \
  -e AERIAL_SVD_GIT_COMMIT="$(git rev-parse HEAD)" \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=2 \
  -v "$PWD:/workspace" -w /workspace aerial-svd-jax:cpu \
  python -m aerial_svd.lowrank_factor_checks \
  --config configs/aid_vit_small.yaml \
  --checkpoint checkpoints/jax/vit_small_protocol_v1.npz \
  --output-dir experiments/lowrank_2026/protocol_v1_cpu_20261117/01_factor_checks \
  --scope all_blocks --matrices blocks.0.attn.proj blocks.0.mlp.fc1 \
  --rank-ratios 0.5 0.75 1.0
```

Выходы: `factor_checks.json`, `checked_factors.npz`, `manifest.json`. Используется
тот же `jax_compression.compress`, что и обычной командой сжатия, с явным списком
целочисленных рангов. Ни одна выбранная матрица не пропускается из-за роста числа
коэффициентов. Проверяются имена, ранги, формы, неизменность bias, соответствие
произведению U_r S_r Vh_r, оптимальная ошибка Фробениуса и полное восстановление.
Полный ранг добавляется как `control=true`, если не запрошен явно.

Нормированный допуск для float32 — `5e-5` (epsilon около `1.19e-7`, SVD и умножение
факторов шириной 384). Фактические ошибки сохраняются; допуск во время проверки не
подбирается. Повторная SVD-референс использует ту же float32-конвенцию: при кратных
сингулярных значениях разные решатели могут выбрать разные оптимальные подпространства.
Тесты дополнительно сравнивают с независимой float64 SVD на некратных спектрах.

NPZ — диагностический архив факторов, **не** candidate checkpoint модели.
Ключи `check_N|<module>.first.weight`, `check_N|<module>.second.weight`,
`check_N|<module>.second.bias` соответствуют строке N в JSON. После записи каждый
массив перечитывается без pickle и сравнивается побитово. Полный native-checkpoint
roundtrip и контроль логитов при полном ранге всех 48 block-проекций покрыты тестами.

## 02_matrix_curves

```bash
docker run --rm --user "$(id -u):$(id -g)" --network none \
  -e AERIAL_SVD_GIT_COMMIT="$(git rev-parse HEAD)" \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=2 \
  -v "$PWD:/workspace" -w /workspace aerial-svd-jax:cpu \
  python -m aerial_svd.lowrank_matrix_curves \
  --config configs/aid_vit_small.yaml \
  --checkpoint checkpoints/jax/vit_small_protocol_v1.npz \
  --output-dir experiments/lowrank_2026/protocol_v1_cpu_20261117/02_matrix_curves \
  --scope all_blocks --matrices blocks.0.attn.proj blocks.0.mlp.fc1 \
  --rank-ratios 0.25 0.5 0.75 1.0
```

Выходы: `matrix_curves.csv`, `manifest.json`. Модель не исполняется, изображения не
открываются; каталог AID может отсутствовать. Требуется существующий split для
записи его идентичности, но индексы не используются для вычисления матричных ошибок.
Один точный спектр на матрицу переиспользуется для всех рангов через существующую
`jax_matrix_analysis.approximation_records`. Ошибки аналитические, по хвосту спектра,
с накоплением энергии в float64; при полном ранге равны нулю в точной арифметике.
Это не измеренная ошибка материализованных факторов (её проверяет этап 01).

CSV: `name`, `m`, `n` (stored out/in), `requested_rank`, `requested_ratio`,
`actual_rank`, `relative_frobenius_error`, `relative_spectral_error`,
`original_coefficients`, `factor_coefficients`, `coefficient_savings`,
`checkpoint_sha256`, `checkpoint`, `method`, `dtype`, `svd_seconds`.
Числа коэффициентов не включают bias. Отрицательная экономия сохраняется.
`svd_seconds` — синхронизированный cold-вызов спектра, включая dispatch/transfer,
одинаков для рангов одной матрицы; это не GPU-бенчмарк и не время инференса.

Для обоих матричных этапов `--scope {ffn,attn,all_blocks,all}` задаёт область;
`--matrices` — необязательный точный подсписок. Без него выбирается вся область.
Вместо `--rank-ratios` можно передать `--ranks 96 192 384` (целые допустимые ранги,
одинаковый список для каждой матрицы). Доли округляются вниз, минимум 1.
Неверный ранг, неизвестное/дублированное имя, имя вне scope — ошибка.
`all_blocks` не включает head; старый `all` сохраняет head.

## Идентификация, сбои и совместимость

Manifest содержит commit, хэши исходников, checkpoint/config/split, исходные
метаданные, архитектуру, arguments, точный список матриц/рангов, dtype, метод,
JAX/NumPy/Python/device. Baseline дополнительно фиксирует классы, предобработку и
состав выборок. При успехе `status=complete` и хэши выходных файлов. Во время работы
`running`, при исключении в вычислении `failed` с причиной. Ошибка preflight не
создаёт каталог. Файлы публикуются атомарно; существующие каталоги не принимаются.
Для восстановления повторяйте только нужный этап в новом каталоге. Сбой процесса
может оставить `running`; это не завершённый результат.

Обычные JAX `compress`, `evaluate`, `finetune_compressed` понимают `all_blocks`.
Новые JAX checkpoint помечаются `all_selected_v1`. Чтение исторических checkpoint
с прежним пропуском при проверке явного scope/rank требует
`--legacy-skip-non-saving` на старых `evaluate`/`finetune_compressed`; такие результаты
помечаются `legacy_skip_non_saving`. Флаг не влияет на создание новых факторизаций.
Без явной идентификации историческая политика помечается как неизвестная.
Исходный Torch `svd.py` сохраняет прежний skip как путь совместимости; использовать
его для нового протокола нельзя. Формат NPZ, bias, split и старые файлы не менялись.

Старая JAX `evaluate` по-прежнему оценивает test; для этапа 00 используйте только
`lowrank_baseline`. `layer_error_analysis` также пока использует test — это
не реализованный здесь этап 03 и не источник для выбора рангов.

## Проверки

В том же CPU-образе с теми же Docker-параметрами:

```bash
python -m unittest discover -s tests -p test_lowrank_stages.py -v
python -m unittest discover -s tests -p test_jax_cli.py -v
```

Сравнительные `test_jax_analysis.py` и `test_jax_training.py` требуют контейнера с
Torch и JAX. Точная команда для доступной локальной среды и результаты приведены
в `VALIDATION_RU.md`. Статически на хосте:

```bash
PYTHONPYCACHEPREFIX=/tmp/lowrank-pycache python3 -m py_compile src/aerial_svd/*.py tests/*.py
git diff --check
```

Этапы 03–09, weighted/activation-aware методы, дообучение, подбор по test и полный
перебор конфигураций в эту реализацию не входят.
