# Полноценная AI-система с нуля: пошаговый план + рабочий код

Этот проект показывает, как построить **универсальную AI-систему** на Python: от идеи и архитектуры до обучения, API-развертывания и оптимизации.

---

## 1) Цель ИИ

Ниже варианты назначения AI-системы:

1. **Чат-бот/ассистент** (поддержка пользователей, FAQ, helpdesk).
2. **Анализ данных/классификация** (тональность, спам, категоризация тикетов).
3. **Генерация текста** (маркетинг, резюме, автодополнение).
4. **Голосовой ассистент** (ASR + NLU + TTS).
5. **Компьютерное зрение** (классификация изображений, детекция).

### Выбор наиболее универсального варианта

Для старта выбираем: **текстовый AI-ассистент с API**, который умеет классифицировать запросы по намерениям (intent classification).

Почему это универсально:
- легко интегрируется в сайт/приложение;
- это база для чат-бота, маршрутизации запросов и NLU;
- архитектуру можно расширить до RAG, генерации и мультимодальности.

---

## 2) Архитектура системы

## 2.1 Компоненты

1. **Data Layer**: сырые тексты + метки intent.
2. **Preprocessing**: токенизация, словарь, паддинг.
3. **Model Layer**: `Embedding -> TransformerEncoder -> Classifier`.
4. **Training Layer**: цикл обучения, валидация, сохранение чекпойнта.
5. **Serving Layer**: FastAPI endpoint `/predict`.
6. **Monitoring**: latency, accuracy, drift по данным.

## 2.2 Модели

- Базовая модель: **Transformer Encoder** (PyTorch).
- Альтернативы:
  - LSTM/GRU (быстрее старт, слабее на длинных контекстах).
  - HuggingFace BERT/RuBERT (лучше качество, тяжелее инференс).

## 2.3 Технологический стек

- **Python 3.10+**
- **PyTorch** — обучение и инференс
- **FastAPI + Uvicorn** — API
- **NumPy** — утилиты
- **(опционально) HuggingFace** — расширение до предобученных LLM

Установка:

```bash
pip install torch fastapi uvicorn numpy
```

---

## 3) Полный рабочий код

См. файл [`ai_system.py`](ai_system.py). Он включает:
- создание игрушечного датасета;
- подготовку словаря;
- определение Transformer-модели;
- обучение;
- сохранение/загрузка;
- API-сервер FastAPI.

### Быстрый запуск

1. Обучить модель:

```bash
python ai_system.py train
```

2. Запустить API:

```bash
python ai_system.py serve
```

3. Тест запроса:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Где мой заказ?"}'
```

---

## 4) Как работает обучение

1. Модель получает батч токенов и маску паддинга.
2. `Embedding` превращает токены в вектора.
3. `TransformerEncoder` извлекает контекст.
4. Mean pooling формирует вектор текста.
5. `Linear` слой предсказывает класс intent.
6. `CrossEntropyLoss` считает ошибку.
7. Adam оптимизатор обновляет веса.

### Что улучшает точность

- увеличить и очистить датасет;
- добавить аугментации текста (синонимы, опечатки, paraphrase);
- перейти на предобученный encoder (RuBERT);
- добавить scheduler (`ReduceLROnPlateau`, cosine);
- использовать раннюю остановку (early stopping);
- балансировать классы (class weights, oversampling).

---

## 5) Развертывание как API + интеграция

## 5.1 Локально

```bash
python ai_system.py serve
```

## 5.2 Через Uvicorn напрямую

```bash
uvicorn ai_system:app --host 0.0.0.0 --port 8000
```

## 5.3 Подключение к фронтенду (пример JS)

```javascript
async function predictIntent(text) {
  const response = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });
  return await response.json();
}
```

---

## 6) Оптимизация, ускорение, масштабирование

### Оптимизация качества

- transfer learning на RuBERT;
- hard-negative mining;
- active learning (доразметка сложных кейсов);
- регулярные offline/online A/B оценки.

### Ускорение инференса

- torchscript / ONNX;
- quantization (int8);
- dynamic batching;
- кэширование частых запросов.

### Масштабирование

- Docker + Kubernetes;
- горизонтальный autoscaling по CPU/RPS;
- очередь (Kafka/RabbitMQ) для асинхронной обработки;
- observability: Prometheus + Grafana + centralized logs.

---

## Структура production-версии

```text
ai-platform/
  data/
  src/
    train.py
    infer.py
    model.py
    api.py
  tests/
  checkpoints/
  docker/
  README.md
```

Если нужно, следующим шагом можно добавить:
- RAG (векторная база + retrieval);
- диалоговый менеджер;
- мультиязычность;
- CI/CD пайплайн и Dockerfile.
