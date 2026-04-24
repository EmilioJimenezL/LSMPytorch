#!/usr/bin/env python3
"""
LSM — Organizador de videos basado en themes.json
===================================================
Lee themes.json, valida cada .mp4 con ffprobe y lo mueve
a la estructura correcta:

    output/
    └── Categoria/
        └── Nombre Palabra/
            └── video.mp4

Archivos inválidos (GIFs, encoding corrupto) → _invalidos/
Archivos sin match en el JSON               → _sin_match/

Uso:
    python3 organize_from_json.py --json themes.json --input ./videos --output ./dataset
    python3 organize_from_json.py --json themes.json --input ./videos --output ./dataset --dry-run
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Forzar stdout a manejar caracteres problemáticos sin crashear
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace', closefd=False)


# ── Validación con ffprobe ────────────────────────────────────────────────────

def is_valid_video(path):
    """
    Retorna (valid: bool, reason: str).
    Usa ffprobe para verificar que el archivo:
      - Tenga al menos un stream de video
      - No sea un GIF renombrado a .mp4
      - Tenga un codec de video real (no image2)
      - Sea decodificable sin errores
    """
    try:
        # Obtener info de streams
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams", "-show_format",
                str(path)
            ],
            capture_output=True, text=True, timeout=15
        )

        if result.returncode != 0:
            return False, "ffprobe error / archivo ilegible"

        info = json.loads(result.stdout)
        streams = info.get("streams", [])
        fmt     = info.get("format", {})

        # Debe tener al menos un stream de video
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        if not video_streams:
            return False, "sin stream de video"

        vs = video_streams[0]
        codec = vs.get("codec_name", "").lower()

        # Rechazar GIFs (codec gif o formato gif)
        format_name = fmt.get("format_name", "").lower()
        if codec == "gif" or "gif" in format_name:
            return False, f"es un GIF (codec={codec})"

        # Rechazar image2 (secuencia de imágenes)
        if codec in ("image2", "png", "mjpeg", "bmp"):
            return False, f"codec de imagen, no video ({codec})"

        # Verificar que tiene duración real
        duration = float(fmt.get("duration", 0))
        if duration < 0.5:
            return False, f"duración muy corta ({duration:.2f}s)"

        # Verificar que tiene frames reales
        nb_frames = vs.get("nb_frames")
        if nb_frames and int(nb_frames) < 5:
            return False, f"muy pocos frames ({nb_frames})"

        return True, "ok"

    except subprocess.TimeoutExpired:
        return False, "timeout al leer"
    except Exception as e:
        return False, f"error: {e}"


# ── Normalización ─────────────────────────────────────────────────────────────

def normalize(text):
    """Minúsculas, sin tildes, sin caracteres especiales, espacios simples."""
    text = text.lower().strip()
    for src, dst in [
        ("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),
        ("ü","u"),("ñ","n"),("à","a"),("è","e"),("ì","i"),
        ("ò","o"),("ù","u"),
    ]:
        text = text.replace(src, dst)
    text = re.sub(r"[-_/\\]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Carga del mapa desde themes.json ─────────────────────────────────────────

def build_word_map(themes_path):
    """
    Construye un diccionario:
        nombre_normalizado → (categoria, carpeta_destino, display_name)

    Solo genera claves desde el videoUrl (fuente de verdad) y el
    display name — sin variantes parciales para evitar falsos positivos.
    """
    with open(themes_path, encoding="utf-8") as f:
        data = json.load(f)

    word_map = {}

    for theme in data:
        categoria = theme["name"]
        for word in theme["words"]:
            display     = word["name"]
            url         = word["videoUrl"]
            dest_folder = url.rstrip("/").split("/")[-1]

            keys = set()

            # 1. Nombre de carpeta completo normalizado
            #    ej. "agua mineral penafiel"
            keys.add(normalize(dest_folder))

            # 2. Sin paréntesis y su contenido
            #    ej. "fanta (color_color_refresco)" → "fanta"
            sin_parentesis = re.sub(r"\(.*?\)", "", dest_folder).strip()
            if sin_parentesis:
                keys.add(normalize(sin_parentesis))

            # 3. Display name normalizado
            keys.add(normalize(display))

            for key in keys:
                if key and key not in word_map:
                    word_map[key] = (categoria, dest_folder, display)

    return word_map


# ── Match de archivo contra el mapa ──────────────────────────────────────────

def find_match(filename_stem, word_map):
    """
    Busca la entrada del mapa que corresponde al archivo.
    Soporta el formato {palabra}[N].mp4 — quita el número final antes de buscar.
    Case insensitive via normalización.
    Retorna (categoria, dest_folder, display_name) o None.
    """
    # Quitar número final: "Abuela2" → "Abuela", "HotDog3" → "HotDog"
    stem_sin_numero = re.sub(r"\d+$", "", filename_stem).strip()

    for candidato in [stem_sin_numero, filename_stem]:
        norm = normalize(candidato)
        if norm in word_map:
            return word_map[norm]

    return None


# ── Organización ──────────────────────────────────────────────────────────────

def organize(themes_path, input_dir, output_dir, dry_run=False, verbose=False):
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    word_map    = build_word_map(themes_path)

    mp4_files = sorted([
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() == ".mp4"
    ])

    if not mp4_files:
        print(f"❌ No se encontraron archivos .mp4 en: {input_dir}")
        return

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Procesando {len(mp4_files)} archivos .mp4")
    print(f"  JSON    : {themes_path}")
    print(f"  Entrada : {input_dir}")
    print(f"  Salida  : {output_dir}\n")

    stats       = {}
    no_match    = []
    replaced    = []
    invalidos   = []   # (nombre, razón)

    for mp4 in mp4_files:

        # ── 0. Filtrar por nombre corrupto ────────────────────────────────
        safe_name = mp4.name.encode("utf-8", errors="replace").decode("utf-8")
        if "(invalid encoding)" in safe_name.lower() or "invalid encoding" in safe_name.lower():
            invalidos.append((safe_name, "nombre con encoding inválido"))
            if verbose:
                print(f"  [INVALIDO  ] {safe_name}  (nombre corrupto)")
            if not dry_run:
                dest = output_path / "_invalidos"
                dest.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(mp4, dest / mp4.name)
                except Exception:
                    pass
            continue

        # ── 1. Validar con ffprobe ────────────────────────────────────────
        valid, reason = is_valid_video(mp4)
        if not valid:
            invalidos.append((mp4.name, reason))
            if verbose:
                print(f"  [INVALIDO  ] {mp4.name}  ({reason})")
            if not dry_run:
                dest = output_path / "_invalidos"
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(mp4, dest / mp4.name)
            continue

        # ── 2. Buscar match en el JSON ────────────────────────────────────
        match = find_match(mp4.stem, word_map)
        if match is None:
            no_match.append(mp4.name)
            if verbose:
                print(f"  [SIN MATCH ] {mp4.name}")
            if not dry_run:
                dest = output_path / "_sin_match"
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(mp4, dest / mp4.name)
            continue

        # ── 3. Mover al destino ───────────────────────────────────────────
        categoria, dest_folder, display = match
        dest_dir  = output_path / categoria / dest_folder
        dest_file = dest_dir / mp4.name

        is_replace = dest_file.exists()
        if is_replace:
            replaced.append(mp4.name)

        stats[categoria] = stats.get(categoria, 0) + 1

        if verbose or dry_run:
            tag = "REEMPLAZAR" if is_replace else "MOVER"
            # Usar encode/decode con reemplazo para nombres con caracteres corruptos
            safe_name = mp4.name.encode("utf-8", errors="replace").decode("utf-8")
            print(f"  [{tag:<10}] {safe_name}")
            print(f"               → {categoria}/{dest_folder}/{safe_name}")

        if not dry_run:
            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(mp4, dest_file)
            except (OSError, ValueError) as e:
                safe_name = mp4.name.encode("utf-8", errors="replace").decode("utf-8")
                print(f"  [ERROR     ] {safe_name}: {e}")
                invalidos.append((safe_name, f"error al copiar: {e}"))
                stats[categoria] -= 1

    # ── Resumen ───────────────────────────────────────────────────────────────
    total = sum(stats.values())
    print(f"\n{'═'*55}")
    print(f"  {'[DRY RUN] ' if dry_run else ''}Resumen")
    print(f"{'═'*55}")

    if stats:
        print(f"\n  Por categoría:")
        for cat, count in sorted(stats.items(), key=lambda x: -x[1]):
            print(f"    {cat:<40} {count:>4} videos")

    print(f"\n  Organizados  : {total}")
    print(f"  Reemplazados : {len(replaced)}")
    print(f"  Inválidos    : {len(invalidos)}")
    print(f"  Sin match    : {len(no_match)}")

    if invalidos:
        print(f"\n  ⚠  Archivos inválidos ({len(invalidos)}):")
        for name, reason in invalidos:
            print(f"     - {name}  ({reason})")
        if not dry_run:
            print(f"  → Copiados a: {output_dir}/_invalidos/")

    if no_match:
        print(f"\n  ⚠  Sin match ({len(no_match)}):")
        for name in no_match:
            print(f"     - {name}")
        if not dry_run:
            print(f"  → Copiados a: {output_dir}/_sin_match/")
        print(f"  → Revísalos manualmente o agrégalos al themes.json.")

    print(f"{'═'*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Organiza videos .mp4 usando themes.json como fuente de verdad"
    )
    parser.add_argument("--json",    required=True, help="Ruta al themes.json")
    parser.add_argument("--input",   required=True, help="Carpeta con los .mp4 planos")
    parser.add_argument("--output",  required=True, help="Carpeta de salida del dataset")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simula sin mover archivos")
    parser.add_argument("--verbose", action="store_true",
                        help="Muestra cada archivo procesado")
    args = parser.parse_args()

    organize(args.json, args.input, args.output,
             dry_run=args.dry_run, verbose=args.verbose)
