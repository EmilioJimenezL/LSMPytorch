#!/usr/bin/env python3
"""
LSM — Lista todas las palabras del themes.json
===============================================
Uso:
    python3 list_words.py --json themes.json
    python3 list_words.py --json themes.json --categoria "Animales e Insectos"
    python3 list_words.py --json themes.json --exportar palabras.txt
"""

import argparse
import json


def list_words(themes_path, categoria=None, exportar=None):
    with open(themes_path, encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    lines = []

    for theme in data:
        cat = theme["name"]

        # Filtrar por categoría si se especificó
        if categoria and categoria.lower() not in cat.lower():
            continue

        words = theme["words"]
        lines.append(f"\n{'─'*50}")
        lines.append(f"  {cat}  ({len(words)} palabras)")
        lines.append(f"{'─'*50}")

        for w in words:
            folder = w["videoUrl"].rstrip("/").split("/")[-1]
            lines.append(f"  {w['name']:<45}  carpeta: {folder}")
            total += 1

    lines.append(f"\n{'═'*50}")
    lines.append(f"  Total: {total} palabras en {len(data)} categorías")
    lines.append(f"{'═'*50}")

    output = "\n".join(lines)
    print(output)

    if exportar:
        with open(exportar, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\n  Exportado a: {exportar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lista todas las palabras del themes.json"
    )
    parser.add_argument("--json",       required=True, help="Ruta al themes.json")
    parser.add_argument("--categoria",  default=None,  help="Filtrar por categoría")
    parser.add_argument("--exportar",   default=None,  help="Guardar resultado en .txt")
    args = parser.parse_args()

    list_words(args.json, args.categoria, args.exportar)
