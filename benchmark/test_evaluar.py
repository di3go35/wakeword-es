#!/usr/bin/env python3
"""Pruebas del banco de evaluación. Todo con fixtures sintéticos y deterministas.

    python3 benchmark/test_evaluar.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evaluar  # noqa: E402
import generar_autotest  # noqa: E402


def reg(tipo, frames, dur_s=None, **extra):
    r = {"id": extra.pop("id", f"{tipo}_1"), "tipo": tipo, "frames": frames}
    if dur_s is not None:
        r["dur_s"] = dur_s
    r.update(extra)
    return r


class TestMetricas(unittest.TestCase):
    def test_eventos_fusiona_por_cooldown(self):
        frames = [(0.00, 0.9), (0.10, 0.9), (0.20, 0.8), (5.00, 0.9)]
        self.assertEqual(evaluar.eventos(frames, 0.5, cooldown_s=1.0), 2)
        # Con cooldown 0 cada frame sobre el umbral es su propio evento.
        self.assertEqual(evaluar.eventos(frames, 0.5, cooldown_s=0.0), 4)
        # Nada supera un umbral alto.
        self.assertEqual(evaluar.eventos(frames, 0.95, cooldown_s=1.0), 0)

    def test_falsos_positivos_por_hora(self):
        neg = [
            reg("negativo", [(0.0, 0.9), (10.0, 0.9)], dur_s=3600.0, id="n1"),
            reg("negativo", [(0.0, 0.1)], dur_s=3600.0, id="n2"),
        ]
        fa_h, horas = evaluar.falsos_positivos_por_hora(neg, 0.5, cooldown_s=1.0)
        self.assertAlmostEqual(horas, 2.0)
        self.assertAlmostEqual(fa_h, 1.0)

    def test_negativos_sin_horas_falla(self):
        with self.assertRaises(ValueError):
            evaluar.falsos_positivos_por_hora([reg("negativo", [(0.0, 0.9)], dur_s=0.0)],
                                              0.5, 1.0)

    def test_recall(self):
        pos = [
            reg("positivo", [(0.0, 0.9)], id="p1"),
            reg("positivo", [(0.0, 0.4)], id="p2"),
            reg("positivo", [(0.0, 0.1)], id="p3"),
            reg("positivo", [(0.0, 0.7)], id="p4"),
        ]
        r = evaluar.recall(pos, 0.5)
        self.assertIsNotNone(r)
        assert r is not None
        # 0.9 y 0.7 superan 0.5; 0.4 y 0.1 no.
        self.assertAlmostEqual(r, 0.5)

    def test_umbral_para_tasa_elige_mayor_recall_y_mas_conservador(self):
        puntos = [
            {"umbral": 0.90, "fa_h": 0.1, "recall": 0.80},
            {"umbral": 0.80, "fa_h": 0.4, "recall": 0.90},
            {"umbral": 0.70, "fa_h": 0.45, "recall": 0.93},
            {"umbral": 0.60, "fa_h": 0.9, "recall": 0.99},
        ]
        elegido = evaluar.umbral_para_tasa(puntos, 0.5)
        self.assertIsNotNone(elegido)
        assert elegido is not None
        self.assertEqual(elegido["umbral"], 0.70)
        self.assertIsNone(evaluar.umbral_para_tasa(puntos, 0.05))


class TestCriterio(unittest.TestCase):
    def test_perfil_bueno_cumple(self):
        informe = evaluar.evaluar(generar_autotest.generar("bueno"))
        self.assertTrue(informe["criterio"]["cumple"], informe["criterio"]["fallos"])
        self.assertEqual(informe["por_objetivo"]["0.5"]["eventos_silencio"], 0)

    def test_perfil_malo_no_cumple(self):
        informe = evaluar.evaluar(generar_autotest.generar("malo"))
        self.assertFalse(informe["criterio"]["cumple"])
        motivos = " ".join(informe["criterio"]["fallos"])
        self.assertIn("recall", motivos)

    def test_una_activacion_en_silencio_basta_para_fallar(self):
        registros = generar_autotest.generar("bueno")
        for r in registros:
            if r["tipo"] == "silencio":
                r["frames"][100][1] = 0.99
        informe = evaluar.evaluar(registros)
        self.assertFalse(informe["criterio"]["cumple"])
        self.assertIn("silencio", " ".join(informe["criterio"]["fallos"]))

    def test_sin_negativos_no_evalua(self):
        registros = [r for r in generar_autotest.generar("bueno") if r["tipo"] != "negativo"]
        with self.assertRaises(ValueError):
            evaluar.evaluar(registros)


class TestManifiesto(unittest.TestCase):
    def test_manifiesto_alterado_se_detecta(self):
        with tempfile.TemporaryDirectory() as d:
            man = Path(d) / "manifiesto.json"
            h = Path(d) / "TEST_HASH"
            man.write_text('{"test": "congelado"}', encoding="utf-8")
            h.write_text(evaluar.hash_manifiesto(man) + "  manifiesto.json\n", encoding="utf-8")
            self.assertTrue(evaluar.verificar_manifiesto(man, h)["coincide"])
            man.write_text('{"test": "manipulado"}', encoding="utf-8")
            self.assertFalse(evaluar.verificar_manifiesto(man, h)["coincide"])


class TestCarga(unittest.TestCase):
    def test_jsonl_invalido_falla_claro(self):
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "p.jsonl"
            ruta.write_text('{"id": "x", "tipo": "negativo", "frames": [[0,0.1]]}\n',
                            encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                evaluar.cargar_jsonl(ruta)
            self.assertIn("dur_s", str(ctx.exception))

            ruta.write_text('{"id": "x", "tipo": "positivo"}\n', encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                evaluar.cargar_jsonl(ruta)
            self.assertIn("sin campos", str(ctx.exception))

    def test_roundtrip_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "p.jsonl"
            with ruta.open("w", encoding="utf-8") as f:
                for r in generar_autotest.generar("bueno"):
                    f.write(json.dumps(r) + "\n")
            cargados = evaluar.cargar_jsonl(ruta)
            self.assertEqual(len(cargados), 60 + 3 + 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
