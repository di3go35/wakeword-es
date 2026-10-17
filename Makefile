PY := python3
TMP := /tmp/wakeword-es

.PHONY: test autotest benchmark limpiar

## Pruebas unitarias del banco de evaluación (fixtures sintéticos).
test:
	$(PY) benchmark/test_evaluar.py

## Autoprueba del criterio: el perfil 'bueno' debe cumplirlo y el 'malo' debe fallarlo.
## Nunca son resultados: son fixtures para verificar la maquinaria.
autotest:
	@mkdir -p $(TMP)
	@echo "--- perfil bueno: debe CUMPLIR el criterio ---"
	$(PY) benchmark/generar_autotest.py --perfil bueno --salida $(TMP)/autotest-bueno.jsonl
	$(PY) benchmark/evaluar.py --puntajes $(TMP)/autotest-bueno.jsonl --criterio
	@echo "--- perfil malo: debe FALLAR (prueba de que el criterio es ejecutable) ---"
	$(PY) benchmark/generar_autotest.py --perfil malo --salida $(TMP)/autotest-malo.jsonl
	@! $(PY) benchmark/evaluar.py --puntajes $(TMP)/autotest-malo.jsonl --criterio && \
		echo "OK: el criterio falló como debía"

## Evalúa el test real. Requiere el test congelado (fase 5) y los puntajes del scorer.
benchmark:
	@test -f benchmark/puntajes.jsonl || \
		{ echo "falta benchmark/puntajes.jsonl: el test congelado se construye en la fase 5"; exit 1; }
	$(PY) benchmark/evaluar.py --puntajes benchmark/puntajes.jsonl \
		--manifiesto benchmark/manifiesto.json --criterio

limpiar:
	rm -rf $(TMP)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
