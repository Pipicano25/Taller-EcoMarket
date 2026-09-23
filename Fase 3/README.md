# EcoMarket + Llama 3.1 8B Instruct

Proyecto para el Taller Práctico #1 de EcoMarket.

## Características

- Token de Hugging Face almacenado en `.env`.
- Prompts separados en archivos `.txt`.
- Contexto de 10 pedidos de prueba.
- Política de devoluciones separada.
- Una única llamada al modelo por cada ejecución.
- Respuestas guardadas automáticamente con fecha, hora, minuto y segundo.

## Instalación

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Crear `.env`:

```powershell
Copy-Item .env.example .env
```

Luego coloca tu token real en `HF_TOKEN`.

## Ejecutar consulta de pedido

```bash
python app.py pedido
```

Usa:
- `prompts/system.txt`
- `prompts/pedido.txt`
- `context/pedidos.txt`

## Ejecutar consulta de devolución

```bash
python app.py devolucion
```

Usa:
- `prompts/system.txt`
- `prompts/devolucion.txt`
- `context/politica_devoluciones.txt`

Cada ejecución realiza exactamente una solicitud al modelo.

Las respuestas quedan almacenadas en:

```text
responses/
```

Ejemplo:

```text
pedido_2026-09-22_21-30-15.txt
devolucion_2026-09-22_21-32-08.txt
```
