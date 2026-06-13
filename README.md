<div align="center">

# Автоулучшение фото

**Команда «Крутые бобры»**

Алексей Исаков (капитан) · Дмитрий Баяндин

</div>

## Цель

Сервис автоматически находит некачественные фото объявлений недвижимости (тёмные, шумные, размытые, пересвеченные, с цветовым сдвигом) и улучшает их ML-моделями. Чистое и понятное фото лучше считывается покупателем и повышает шанс, что объявление дойдёт до контакта с продавцом, при этом сервис не дорисовывает несуществующих деталей (доверие важнее красоты) и укладывается в SLA p99 ≤ 1 с на фото 1080p.

По эвристикам команды улучшения требуют 38 831 из 100 000 фото недвижимости. Улучшение фото должно повышать число покупок на Avito.

## Как работает пайплайн

```mermaid
flowchart TD
    IN([" Вход: фото JPEG "]):::io --> CLF{" Тип кадра?<br/>классификатор MobileNetV3 "}:::clf

    CLF -->|screenshot| CROP[" Срез чёрных полос "]:::step
    CROP --> RECLF{" Тип после среза? "}:::clf
    RECLF --> ASSESS
    CLF -->|" real_estate / floor_plan "| ASSESS[" Оценка качества:<br/>яркость, контраст, резкость, шум "]:::step

    ASSESS --> ROUTE{" Роутер:<br/>метрики + тип кадра "}:::clf

    ROUTE -->|" хорошее фото "| SKIP([" Оригинал без изменений "]):::io
    ROUTE -->|" real_estate "| TONE[" Тон:<br/>Retinexformer / CoTF "]:::model
    ROUTE -->|" floor_plan "| DET
    TONE --> DET[" Детали:<br/>SCUNet / Real-SAFMN++ "]:::model

    DET --> IQA{" IQA-гейт BRISQUE/NIQE:<br/>стало лучше? "}:::clf
    IQA -->|да| OUT([" Выход: улучшенное JPEG<br/>+ заголовки X-Enhance-* "]):::io
    IQA -->|нет| FB([" Fallback: вернуть оригинал "]):::io

    classDef io fill:#1f6feb,color:#fff,stroke:#0b3d91,stroke-width:1px;
    classDef clf fill:#8957e5,color:#fff,stroke:#4b2a8a,stroke-width:1px;
    classDef step fill:#2da44e,color:#fff,stroke:#116329,stroke-width:1px;
    classDef model fill:#bf8700,color:#fff,stroke:#7d5700,stroke-width:1px;
```

## Метрики

**Бизнес-метрики:**

| Метрика               | Что это |
|-----------------------|---|
| conversion_to_contact | целевые действия (звонок, сообщение, избранное, покупка) / просмотры объявления |
| preference rate       | доля выборов «улучшено» в слепом сравнении с оригиналом |
| SLA p99               | время обработки одного фото 1080p |
| RPS                   | пропускная способность сервиса |

**Метрики качества изображения:**

| Метрика | Что измеряет |
|---|---|
| `brightness_mean`, `contrast_std` | яркость и контраст |
| `sharpness_laplacian_var` | резкость (дисперсия лапласиана) |
| `noise_sigma` | уровень шума |
| `entropy` | информативность изображения |
| `PSNR`, `SSIM` | близость к эталону по пикселям и структуре (выше лучше) |
| `LPIPS` | перцептивное отличие от эталона (ниже лучше) |
| `BRISQUE`, `NIQE` | no-reference качество, основа IQA-гейта (ниже лучше) |

## Запуск

```bash
bash infra/deploy/bootstrap.sh   # один раз: docker + nvidia-container-toolkit
make install
make weights
make up-gpu
make logs-gpu
```


## HTML

```bash
services/enhancer/.venv/bin/python scripts/eval_endpoint.py --html data/eval_set/report
```
