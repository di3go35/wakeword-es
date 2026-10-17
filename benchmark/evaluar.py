#!/usr/bin/env python3
"""Banco de evaluación de wakeword-es.

Criterio de éxito (congelado en el plan):

  - >=95% de recall a <=0,5 falsos positivos por hora, en cada condición.
  - 0 activaciones en el test de silencio.
  - El manifiesto del test debe coincidir con el hash congelado.

Formato de entrada (JSONL, un registro por línea):

  {"id": "pos_0001", "tipo": "positivo", "condicion": "limpio",
   "hablante": "h03", "motor_tts": "chatterbox",
   "frames": [[0.0, 0.02], [0.08, 0.31]]}

  {"id": "neg_cv_0007", "tipo": "negativo", "condicion": "-",
   "fuente": "common_voice_es", "dur_s": 600.0,
   "frames": [[0.0, 0.01], [1.0, 0.03]]}

  {"id": "sil_0001", "tipo": "silencio", "condicion": "silencio_digital",
   "dur_s": 1800.0, "frames": [[0.0, 0.0]]}

`frames` son pares [t_segundos, score] con score en [0, 1].

  - tipo "positivo": el clip contiene la frase. Se considera detectado si algún
    frame alcanza el umbral.
  - tipo "negativo": audio continuo sin la frase. Requiere `dur_s` explícito para
    calcular falsos positivos por hora.
  - tipo "silencio": silencio digital. Requiere `dur_s`. Cualquier activación aquí
    es un fallo automático.

Uso:

  python3 benchmark/evaluar.py --puntajes puntajes.jsonl --criterio
  python3 benchmark/evaluar.py --puntajes puntajes.jsonl --json informe.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

TIPOS = ("positivo", "negativo", "silencio")

# Criterio de éxito congelado (no se cambia sin reescribir el plan).
RECALL_MINIMO = 0.95
FA_H_CRITERIO = 0.5
OBJETIVOS_FA_H = (0.2, 0.5, 1.0)
# Tolerancia para comparaciones entre fracciones de clips (19/20 frente a 0.95).
TOLERANCIA = 1e-9


# ---------------------------------------------------------------- carga


def cargar_jsonl(ruta: str | Path) -> list[dict]:
    """Carga y valida el JSONL de puntajes. Falla ruidosamente ante datos ambiguos."""
    registros = []
    for n, linea in enumerate(Path(ruta).read_text(encoding="utf-8").splitlines(), 1):
        linea = linea.strip()
        if not linea:
            continue
        try:
            r = json.loads(linea)
        except json.JSONDecodeError as e:
            raise ValueError(f"{ruta}: línea {n} no es JSON válido: {e}") from e
        faltan = {"id", "tipo", "frames"} - set(r)
        if faltan:
            raise ValueError(f"{ruta}: línea {n} sin campos {sorted(faltan)}")
        if r["tipo"] not in TIPOS:
            raise ValueError(f"{ruta}: línea {n} tipo '{r['tipo']}' no es {TIPOS}")
        frames = [(float(t), float(s)) for t, s in r["frames"]]
        if any(s < 0.0 or s > 1.0 for _, s in frames):
            raise ValueError(f"{ruta}: línea {n} tiene scores fuera de [0, 1]")
        frames.sort(key=lambda p: p[0])
        r["frames"] = frames
        if r["tipo"] in ("negativo", "silencio") and "dur_s" not in r:
            raise ValueError(
                f"{ruta}: línea {n} de tipo '{r['tipo']}' necesita 'dur_s' explícito "
                "para poder contar falsos positivos por hora"
            )
        if "dur_s" in r:
            r["dur_s"] = float(r["dur_s"])
            if r["dur_s"] <= 0:
                raise ValueError(f"{ruta}: línea {n} tiene dur_s <= 0")
        registros.append(r)
    if not registros:
        raise ValueError(f"{ruta}: no contiene registros")
    return registros


# ---------------------------------------------------------------- métricas


def eventos(frames: list[tuple[float, float]], umbral: float, cooldown_s: float) -> int:
    """Cuenta activaciones: cada frame >= umbral es candidato; los que caen dentro del
    cooldown posterior a una activación se fusionan (debounce)."""
    total = 0
    ultimo = None
    for t, score in frames:
        if score >= umbral:
            if ultimo is None or (t - ultimo) >= cooldown_s:
                total += 1
                ultimo = t
    return total


def falsos_positivos_por_hora(
    registros: list[dict], umbral: float, cooldown_s: float
) -> tuple[float, float]:
    """Devuelve (falsos positivos por hora, horas totales)."""
    segundos = sum(r["dur_s"] for r in registros)
    if segundos <= 0:
        raise ValueError("los registros negativos suman 0 horas")
    horas = segundos / 3600.0
    total = sum(eventos(r["frames"], umbral, cooldown_s) for r in registros)
    return total / horas, horas


def recall(registros: list[dict], umbral: float) -> float | None:
    """Fracción de clips positivos con al menos un frame >= umbral."""
    if not registros:
        return None
    detectados = sum(1 for r in registros if any(s >= umbral for _, s in r["frames"]))
    return detectados / len(registros)


def recall_por_grupo(registros: list[dict], umbral: float, campo: str) -> dict:
    """Recall desglosado por un campo (condicion, hablante, motor_tts, ...)."""
    grupos: dict[str, list[int]] = {}
    for r in registros:
        clave = str(r.get(campo, "-"))
        valor = 1 if any(s >= umbral for _, s in r["frames"]) else 0
        grupos.setdefault(clave, []).append(valor)
    return {k: sum(v) / len(v) for k, v in sorted(grupos.items())}


def curva(registros_pos: list[dict], registros_neg: list[dict], umbrales, cooldown_s):
    """Puntos (umbral, fa_h, recall) de mayor a menor umbral."""
    return [
        {
            "umbral": u,
            "fa_h": falsos_positivos_por_hora(registros_neg, u, cooldown_s)[0],
            "recall": recall(registros_pos, u),
        }
        for u in umbrales
    ]


def umbral_para_tasa(puntos: list[dict], objetivo_fa_h: float) -> dict | None:
    """El umbral con mayor recall cuyo FA/h no supera el objetivo.

    Empate: se prefiere el umbral más alto (más conservador).
    """
    candidatos = [p for p in puntos if p["fa_h"] <= objetivo_fa_h and p["recall"] is not None]
    if not candidatos:
        return None
    return max(candidatos, key=lambda p: (p["recall"], p["umbral"]))


# ---------------------------------------------------------------- evaluación


def umbrales_por_defecto() -> list[float]:
    return [round(0.99 - 0.01 * i, 2) for i in range(99)]  # 0.99 .. 0.01


def evaluar(
    registros: list[dict],
    objetivos=OBJETIVOS_FA_H,
    cooldown_s: float = 1.0,
    umbrales=None,
    recall_minimo: float = RECALL_MINIMO,
    fa_h_criterio: float = FA_H_CRITERIO,
) -> dict:
    """Evalúa la batería y devuelve el informe. No imprime nada."""
    pos = [r for r in registros if r["tipo"] == "positivo"]
    neg = [r for r in registros if r["tipo"] == "negativo"]
    sil = [r for r in registros if r["tipo"] == "silencio"]
    if not pos:
        raise ValueError("no hay registros positivos: el recall no sería calculable")
    if not neg:
        raise ValueError("no hay registros negativos: FA/h no sería calculable")

    puntos = curva(pos, neg, umbrales or umbrales_por_defecto(), cooldown_s)
    horas_neg = sum(r["dur_s"] for r in neg) / 3600.0

    por_objetivo = {}
    for obj in objetivos:
        elegido = umbral_para_tasa(puntos, obj)
        clave = f"{obj:.1f}"
        if elegido is None:
            por_objetivo[clave] = {"alcanzable": False}
            continue
        por_objetivo[clave] = {
            "alcanzable": True,
            "umbral": elegido["umbral"],
            "fa_h": elegido["fa_h"],
            "recall": elegido["recall"],
            "recall_por_condicion": recall_por_grupo(pos, elegido["umbral"], "condicion"),
            "eventos_silencio": sum(
                eventos(r["frames"], elegido["umbral"], cooldown_s) for r in sil
            ),
        }

    informe = {
        "cooldown_s": cooldown_s,
        "horas_negativas": horas_neg,
        "n_positivos": len(pos),
        "n_negativos": len(neg),
        "n_silencio": len(sil),
        "det": puntos,
        "por_objetivo": por_objetivo,
    }
    informe["criterio"] = veredicto(informe, recall_minimo, fa_h_criterio)
    return informe


def veredicto(
    informe: dict, recall_minimo: float = RECALL_MINIMO, fa_h_criterio: float = FA_H_CRITERIO
) -> dict:
    """Convierte el criterio de éxito en un veredicto ejecutable."""
    clave = f"{fa_h_criterio:.1f}"
    fallos = []
    objetivo = informe["por_objetivo"].get(clave)
    if not objetivo or not objetivo.get("alcanzable"):
        fallos.append(
            f"ningún umbral alcanza {fa_h_criterio} FA/h: el modelo no llega al criterio"
        )
    else:
        if objetivo["eventos_silencio"] != 0:
            fallos.append(
                f"{objetivo['eventos_silencio']} activaciones en silencio digital "
                "(el criterio exige 0)"
            )
        if objetivo["recall"] < recall_minimo - TOLERANCIA:
            fallos.append(
                f"recall global {objetivo['recall']:.3f} < {recall_minimo:.2f} "
                f"a {objetivo['fa_h']:.3f} FA/h (umbral {objetivo['umbral']})"
            )
        for condicion, valor in objetivo["recall_por_condicion"].items():
            if valor < recall_minimo - TOLERANCIA:
                fallos.append(
                    f"recall en condición '{condicion}' {valor:.3f} < {recall_minimo:.2f}"
                )
    return {"cumple": not fallos, "fallos": fallos}


# ---------------------------------------------------------------- manifiesto


def hash_manifiesto(ruta: str | Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def verificar_manifiesto(ruta_manifiesto: str | Path, ruta_hash: str | Path) -> dict:
    """El test está congelado si el hash del manifiesto coincide con el publicado."""
    actual = hash_manifiesto(ruta_manifiesto)
    esperado = Path(ruta_hash).read_text(encoding="utf-8").split()[0].strip()
    return {"actual": actual, "esperado": esperado, "coincide": actual == esperado}


# ---------------------------------------------------------------- informe legible


def imprimir(informe: dict) -> None:
    print(f"registros: {informe['n_positivos']} positivos, "
          f"{informe['n_negativos']} negativos, {informe['n_silencio']} de silencio")
    print(f"horas negativas: {informe['horas_negativas']:.3f} "
          f"(cooldown {informe['cooldown_s']} s)")
    print()
    print(f"{'objetivo':>9} {'umbral':>7} {'FA/h':>7} {'recall':>7} {'silencio':>9}")
    for clave, o in sorted(informe["por_objetivo"].items(), key=lambda kv: float(kv[0])):
        if not o.get("alcanzable"):
            print(f"{clave:>9} {'-':>7} {'-':>7} {'no alcanzable':>7} {'-':>9}")
            continue
        print(f"{clave:>9} {o['umbral']:>7.2f} {o['fa_h']:>7.3f} "
              f"{o['recall']:>7.3f} {o['eventos_silencio']:>9d}")
        for condicion, valor in o["recall_por_condicion"].items():
            print(f"{'':>9}   · {condicion:<20} recall {valor:.3f}")
    print()
    c = informe["criterio"]
    print(f"CRITERIO: {'CUMPLE' if c['cumple'] else 'NO CUMPLE'}")
    for f in c["fallos"]:
        print(f"  - {f}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Banco de evaluación de wakeword-es")
    p.add_argument("--puntajes", required=True, help="JSONL con los puntajes por frame")
    p.add_argument("--cooldown", type=float, default=1.0,
                   help="segundos de debounce para contar activaciones (por defecto 1.0)")
    p.add_argument("--manifiesto", help="manifiesto del test (para verificar el congelado)")
    p.add_argument("--hash", dest="ruta_hash",
                   default=str(Path(__file__).parent / "TEST_HASH"),
                   help="archivo con el hash congelado del manifiesto")
    p.add_argument("--json", help="escribe el informe completo en este archivo")
    p.add_argument("--criterio", action="store_true",
                   help="devuelve código de salida 1 si el criterio no se cumple")
    args = p.parse_args(argv)

    if args.manifiesto and args.ruta_hash and Path(args.ruta_hash).exists():
        v = verificar_manifiesto(args.manifiesto, args.ruta_hash)
        print(f"manifiesto: {v['actual'][:12]} {'OK' if v['coincide'] else 'ALTERADO'}"
              f" (esperado {v['esperado'][:12]})")
        if not v["coincide"]:
            print("El test ya no es el que se congeló. Se aborta la evaluación.")
            return 2
    else:
        print("manifiesto: no verificado (sin --manifiesto o sin hash congelado)")

    informe = evaluar(cargar_jsonl(args.puntajes), cooldown_s=args.cooldown)
    imprimir(informe)

    if args.json:
        Path(args.json).write_text(
            json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\ninforme escrito en {args.json}")

    if args.criterio and not informe["criterio"]["cumple"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
