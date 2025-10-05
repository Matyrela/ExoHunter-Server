# ---- build ----
FROM python:3.12-slim AS build
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# 🔧 Dependencias necesarias para compilar paquetes (solo en build)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip wheel --no-cache-dir --no-deps -r requirements.txt -w /wheels
COPY . .

# ---- runtime ----
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# ⚙️ Instalar librerías del sistema requeridas por lightgbm y psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# 🔽 Copiar los wheels y dependencias
COPY --from=build /wheels /wheels
RUN pip install --no-cache /wheels/*

# Copiar el código fuente
COPY . .

# Puerto típico de FastAPI/uvicorn
EXPOSE 8000

# 🛰️ Ejecutar la app (ajusta el path si es necesario)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
