#!/bin/bash
# Script de arranque para Azure App Service Linux
# Azure asigna el puerto via $PORT (típicamente 8000)

# Esperar a que el sistema esté listo
echo "Arrancando Sign.Bridge en puerto ${PORT:-8000}…"

# Ejecutar con uvicorn directamente (más confiable que python main.py)
gunicorn main:app \
    --bind 0.0.0.0:${PORT:-8000} \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 600 \
    --access-logfile - \
    --error-logfile -
