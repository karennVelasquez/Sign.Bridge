"""
╔══════════════════════════════════════════════════════════════╗
║          SignLens — Traductor de Señas en Tiempo Real        ║
║          CNN + MediaPipe | TensorFlow/Keras | Python         ║
╚══════════════════════════════════════════════════════════════╝

Uso:
    python main.py                  → Modo traducción (inference)
    python main.py --train          → Modo entrenamiento
    python main.py --collect HOLA   → Recolectar datos para "HOLA"
    python main.py --list           → Listar palabras entrenadas
    python main.py --delete HOLA    → Eliminar una palabra

Controles en ventana:
    Q  → Salir
    E  → Empezar/detener captura de muestras
    T  → Entrenar modelo (desde modo collect)
    C  → Limpiar pantalla de predicciones
"""

import argparse
import sys
from app.collector import DataCollector
from app.trainer import ModelTrainer
from app.translator import RealTimeTranslator
from utils.storage import DataStorage


def main():
    parser = argparse.ArgumentParser(
        description="SignLens - Traductor de Lenguaje de Señas CNN"
    )
    parser.add_argument("--train",    action="store_true", help="Entrenar el modelo con datos recolectados")
    parser.add_argument("--collect",  type=str, metavar="PALABRA", help="Recolectar muestras para una palabra")
    parser.add_argument("--list",     action="store_true", help="Listar todas las palabras entrenadas")
    parser.add_argument("--delete",   type=str, metavar="PALABRA", help="Eliminar datos de una palabra")
    parser.add_argument("--samples",  type=int, default=200, help="Número de muestras por sesión (default: 200)")
    parser.add_argument("--epochs",   type=int, default=50, help="Épocas de entrenamiento (default: 50)")
    parser.add_argument("--camera",   type=int, default=0, help="Índice de cámara (default: 0)")

    args = parser.parse_args()
    storage = DataStorage()

    if args.list:
        words = storage.list_words()
        if not words:
            print("\n⚠️  No hay palabras entrenadas aún.")
            print("   Usa: python main.py --collect HOLA\n")
        else:
            print(f"\n📚 Palabras registradas ({len(words)}):")
            for w in words:
                count = storage.count_samples(w)
                print(f"   ✦ {w:<20} {count} muestras")
            print()
        return

    if args.delete:
        word = args.delete.upper()
        if storage.delete_word(word):
            print(f"\n🗑️  Datos de '{word}' eliminados.\n")
        else:
            print(f"\n⚠️  No se encontraron datos para '{word}'.\n")
        return

    if args.collect:
        word = args.collect.upper()
        print(f"\n📷 Iniciando recolección para: '{word}'")
        print(f"   Muestras objetivo: {args.samples}")
        print(f"   Presiona [E] para empezar/pausar | [Q] para salir\n")
        collector = DataCollector(camera_index=args.camera)
        collector.collect(word=word, target_samples=args.samples)
        return

    if args.train:
        words = storage.list_words()
        if len(words) < 2:
            print(f"\n⚠️  Necesitas al menos 2 palabras para entrenar.")
            print(f"   Actualmente tienes: {words if words else 'ninguna'}")
            print(f"   Usa: python main.py --collect PALABRA\n")
            sys.exit(1)
        print(f"\n🧠 Iniciando entrenamiento CNN...")
        print(f"   Palabras: {words}")
        print(f"   Épocas: {args.epochs}\n")
        trainer = ModelTrainer()
        trainer.train(epochs=args.epochs)
        return

    # Default: Modo traducción en tiempo real
    if not storage.model_exists():
        print("\n⚠️  No hay modelo entrenado.")
        words = storage.list_words()
        if len(words) >= 2:
            print(f"   Tienes datos para: {words}")
            print(f"   Entrena primero: python main.py --train\n")
        else:
            print(f"   Primero recolecta datos: python main.py --collect HOLA")
            print(f"   Luego entrena: python main.py --train\n")
        sys.exit(1)

    print("\n🖐️  SignLens — Modo Traducción en Tiempo Real")
    print("   Presiona [Q] para salir | [C] para limpiar historial\n")
    translator = RealTimeTranslator(camera_index=args.camera)
    translator.run()


if __name__ == "__main__":
    main()
