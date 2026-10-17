#!/usr/bin/env python3
"""Pruebas del planificador de lotes. Solo biblioteca estándar.

    python3 pipeline/test_planificar_lotes.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import planificar_lotes as pl  # noqa: E402

CABECERA = ["client_id", "path", "sentence", "up_votes", "down_votes", "age", "gender",
            "accents", "variant", "locale", "segment", "duration"]


def escribir_tsv(ruta: Path, filas: list[list[str]], cabecera=None) -> Path:
    cabecera = cabecera or CABECERA
    ruta.write_text(
        "\t".join(cabecera) + "\n"
        + "\n".join("\t".join(f) for f in filas) + "\n",
        encoding="utf-8",
    )
    return ruta


def fila(i: int, acento: str, dur: str = "3600") -> list[str]:
    return [f"cl{i}", f"clip_{i:03d}.mp3", "hola", "3", "0", "", "", acento, "",
            "es", f"seg{i}", dur]


class TestPlanificador(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reparte_por_horas(self):
        ruta = escribir_tsv(self.dir / "validated.tsv",
                            [fila(i, "andino" if i < 3 else "mexicano") for i in range(6)])
        filas, columnas = pl.leer_tsv(ruta)
        dur, modo = pl.duraciones(filas, columnas, pl.DURACION_MEDIA_POR_DEFECTO)
        self.assertEqual(modo, "tsv")
        lotes = pl.planificar(filas, dur, horas_por_lote=2.0)
        # cada clip dura 1 h: los lotes de 2 h llevan 2 clips
        self.assertEqual([len(l["clips"]) for l in lotes], [2, 2, 2])
        self.assertEqual([l["horas"] for l in lotes], [2.0, 2.0, 2.0])

    def test_filtro_de_acento(self):
        ruta = escribir_tsv(self.dir / "validated.tsv",
                            [fila(i, "andino" if i < 3 else "mexicano") for i in range(6)])
        filas, columnas = pl.leer_tsv(ruta)
        dur, _ = pl.duraciones(filas, columnas, pl.DURACION_MEDIA_POR_DEFECTO)
        lotes = pl.planificar(filas, dur, horas_por_lote=100, acentos={"andino"})
        self.assertEqual(sum(len(l["clips"]) for l in lotes), 3)
        self.assertTrue(all(c["accents"] == "andino"
                            for l in lotes for c in l["clips"]))

    def test_sin_columna_duration_usa_estimacion_declarada(self):
        cabecera = [c for c in CABECERA if c != "duration"]
        filas_sin = [f[:-1] for f in [fila(i, "andino") for i in range(4)]]
        ruta = escribir_tsv(self.dir / "validated.tsv", filas_sin, cabecera)
        filas, columnas = pl.leer_tsv(ruta)
        dur, modo = pl.duraciones(filas, columnas, 5.5)
        self.assertEqual(modo, "estimada")
        self.assertAlmostEqual(sum(dur), 4 * 5.5)

    def test_duraciones_mixtas_se_reportan(self):
        ruta = escribir_tsv(self.dir / "validated.tsv",
                            [fila(0, "andino", "10"), fila(1, "andino", "")])
        filas, columnas = pl.leer_tsv(ruta)
        dur, modo = pl.duraciones(filas, columnas, 5.5)
        self.assertIn("1/2", modo)
        self.assertAlmostEqual(sum(dur), 15.5)

    def test_tsv_sin_path_falla_claro(self):
        ruta = self.dir / "otro.tsv"
        ruta.write_text("a\tb\n1\t2\n", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            pl.leer_tsv(ruta)
        self.assertIn("Common Voice", str(ctx.exception))

    def test_escribe_manifiesto_y_listas(self):
        ruta = escribir_tsv(self.dir / "validated.tsv", [fila(i, "andino") for i in range(3)])
        filas, columnas = pl.leer_tsv(ruta)
        dur, modo = pl.duraciones(filas, columnas, 5.5)
        lotes = pl.planificar(filas, dur, horas_por_lote=1.0)
        m = pl.escribir(lotes, self.dir / "lotes", modo)
        self.assertEqual(m["total_clips"], 3)
        self.assertTrue((self.dir / "lotes" / "manifiesto_lotes.json").exists())
        lista = (self.dir / "lotes" / "lote_000.txt").read_text().splitlines()
        self.assertEqual(lista, ["clip_000.mp3"])
        self.assertIn("por_acento", m["lotes"][0])

    def test_horas_por_lote_invalido(self):
        with self.assertRaises(ValueError):
            pl.planificar([{"path": "a.mp3", "accents": "andino"}], [1.0], horas_por_lote=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
