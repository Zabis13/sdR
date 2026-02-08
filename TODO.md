# sdR TODO

## Архитектура

**Подход 2: Rcpp обёртки вокруг C++ stable-diffusion.cpp**

- C++ (src/sd/): токенизаторы, энкодеры, денойзер, семплер, VAE, загрузка моделей
- R: параметризация, хелперы, тестирование, высокоуровневый API
- ggmlR: libggml.a + ggml хедеры через LinkingTo

## Сделано

- [x] C++ исходники stable-diffusion.cpp скопированы в src/sd/
- [x] src/sdR_interface.cpp — Rcpp обёртки (XPtr с кастомным finalizer для sd_ctx_t, upscaler_ctx_t)
- [x] src/Makevars — сборка sd/*.cpp, линковка с libggml.a, -DGGML_MAX_NAME=128
- [x] R/pipeline.R — sd_ctx(), sd_txt2img(), sd_img2img()
- [x] R/zzz.R — константы (0-based, совпадают с C++ enum), .onLoad → sd_init_log()
- [x] R/utils.R — sd_system_info() через C++
- [x] R/image_utils.R — sd_save_image(), sd_tensor_to_image(), sd_image_to_tensor()
- [x] R/sdR-package.R — useDynLib, importFrom Rcpp
- [x] DESCRIPTION — Rcpp, LinkingTo: Rcpp + ggmlR
- [x] Удалены 18 R файлов (чистые R реализации моделей/слоёв)
- [x] Добавлены r_ggml_compat.h и ggml-vulkan.h в ggmlR inst/include, ggmlR переустановлен
- [x] Makevars: использует installed ggmlR через LinkingTo, -include r_ggml_compat.h
- [x] Компиляция и установка sdR — OK
- [x] NAMESPACE обновлён через roxygen2
- [x] library(sdR) загружается, sd_system_info() работает
- [x] pipeline.R работает с реальной моделью SD 1.5
- [x] XPtr корректно создаётся/уничтожается
- [x] sd_txt2img() — генерация 512x512 за ~7с (Vulkan GPU)
- [x] sd_save_image() — сохранение PNG
- [x] Vulkan бэкенд работает (radv, AMD GPU)

## Осталось

### 1. [ ] Вынести vocab*.hpp из пакета (128 МБ) — БЛОКЕР для CRAN
- vocab.hpp (29 MB) — CLIP токенизатор (SD 1.x, SD 2.x, SDXL)
- vocab_mistral.hpp (35 MB) — Mistral текст-энкодер
- vocab_qwen.hpp (10 MB) — Qwen текст-энкодер
- vocab_umt5.hpp (54 MB) — UMT5 токенизатор (SD3, Flux)
- Найти источник оригинальных данных (upstream sd.cpp / HuggingFace)
- Реализовать скачивание при configure или первом запуске
- CRAN лимит на tarball: 5 МБ

### 2. [ ] Уменьшить размер пакета до CRAN лимита (5 МБ)
- Проверить размер без vocab файлов
- Убедиться что остальные исходники (ggml, sd.cpp) укладываются

### 3. [ ] Настроить configure скрипт для скачивания зависимостей
- Скачивание vocab файлов при установке
- Fallback и информативные сообщения об ошибках

### 4. [ ] Подготовить DESCRIPTION и документацию для CRAN
- License, SystemRequirements, URL, BugReports
- Документация всех экспортируемых функций
- Vignettes

### 5. [ ] R CMD check --as-cran
- Зависит от задач 1–4
- 0 errors, 0 warnings, 0 notes

### 6. [ ] Предупреждения компиляции (косметика)
- GGML_ATTRIBUTE_FORMAT redefined — r_ggml_compat.h vs ggml.h
- SSE/AVX = 0 в sd_system_info — ggml SIMD определяется при сборке ggmlR, не sdR

### 7. [ ] Доработать функционал
- Проверить img2img, upscaler
- Конвертация изображений (sd_image_t ↔ R raw vector)

## Ключевые файлы

| Файл | Назначение |
|------|-----------|
| src/sd/stable-diffusion.h | Публичный C API |
| src/sd/stable-diffusion.cpp | Главная реализация (~3800 строк) |
| src/sd/model.cpp | Загрузка моделей (safetensors, gguf) |
| src/sd/ggml_extend.hpp | Мост SD ↔ ggml |
| src/sdR_interface.cpp | Rcpp обёртки с XPtr |
| src/Makevars | Сборка: sd/*.cpp + libggml.a |
| R/pipeline.R | Пользовательский API |
| R/zzz.R | Константы, .onLoad |
| R/image_utils.R | I/O изображений |
