import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
HF_PROVIDER = os.getenv("HF_PROVIDER", "auto")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"
CONTEXT_DIR = BASE_DIR / "context"
RESPONSES_DIR = BASE_DIR / "responses"


CONFIGURACIONES = {
    "pedido": {
        "prompt": PROMPTS_DIR / "pedido.txt",
        "context": CONTEXT_DIR / "pedidos.txt",
        "placeholder": "{{PEDIDOS}}",
    },
    "devolucion": {
        "prompt": PROMPTS_DIR / "devolucion.txt",
        "context": CONTEXT_DIR / "politica_devoluciones.txt",
        "placeholder": "{{POLITICA_DEVOLUCIONES}}",
    },
}


def leer_txt(ruta: Path) -> str:
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo: {ruta}")
    contenido = ruta.read_text(encoding="utf-8").strip()
    if not contenido:
        raise ValueError(f"El archivo está vacío: {ruta}")
    return contenido


def obtener_valor(objeto, atributo, default="No disponible"):
    try:
        valor = getattr(objeto, atributo, default)
        return default if valor is None else valor
    except Exception:
        return default


def construir_prompt(tipo: str) -> tuple[str, str]:
    config = CONFIGURACIONES[tipo]

    prompt = leer_txt(config["prompt"])
    contexto = leer_txt(config["context"])

    if config["placeholder"] not in prompt:
        raise ValueError(
            f"El prompt {config['prompt'].name} no contiene "
            f"el marcador {config['placeholder']}."
        )

    prompt_final = prompt.replace(config["placeholder"], contexto)
    return prompt_final, contexto


def guardar_respuesta(tipo, system_prompt, user_prompt, respuesta, response):
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

    ahora = datetime.now()
    timestamp = ahora.strftime("%Y-%m-%d_%H-%M-%S")
    archivo = RESPONSES_DIR / f"{tipo}_{timestamp}.txt"

    usage = obtener_valor(response, "usage", None)

    if usage:
        prompt_tokens = obtener_valor(usage, "prompt_tokens")
        completion_tokens = obtener_valor(usage, "completion_tokens")
        total_tokens = obtener_valor(usage, "total_tokens")
    else:
        prompt_tokens = completion_tokens = total_tokens = "No disponible"

    contenido = f"""FECHA Y HORA
============
{ahora.strftime("%Y-%m-%d %H:%M:%S")}

TIPO DE SOLICITUD
=================
{tipo}

MODELO
======
{HF_MODEL}

PROVEEDOR
=========
{HF_PROVIDER}

CONSUMO DE TOKENS
=================
Tokens de entrada : {prompt_tokens}
Tokens de salida  : {completion_tokens}
Tokens totales    : {total_tokens}

SYSTEM PROMPT
=============
{system_prompt}

PROMPT ENVIADO AL MODELO
========================
{user_prompt}

RESPUESTA
=========
{respuesta}
"""

    archivo.write_text(contenido, encoding="utf-8")
    return archivo


def main():
    if not HF_TOKEN:
        raise ValueError(
            "No se encontró HF_TOKEN. Crea un archivo .env a partir de .env.example."
        )

    if len(sys.argv) != 2 or sys.argv[1].lower() not in CONFIGURACIONES:
        opciones = " | ".join(CONFIGURACIONES.keys())
        print(f"Uso: python app.py [{opciones}]")
        print("Ejemplo: python app.py pedido")
        raise SystemExit(1)

    tipo = sys.argv[1].lower()

    system_prompt = leer_txt(PROMPTS_DIR / "system.txt")
    user_prompt, _ = construir_prompt(tipo)

    client = InferenceClient(
        provider=HF_PROVIDER,
        api_key=HF_TOKEN,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    print("=" * 65)
    print("EcoMarket - Enviando UNA única solicitud")
    print(f"Tipo      : {tipo}")
    print(f"Modelo    : {HF_MODEL}")
    print(f"Proveedor : {HF_PROVIDER}")
    print("=" * 65)

    # UNA SOLA LLAMADA AL MODELO POR CADA EJECUCIÓN
    response = client.chat.completions.create(
        model=HF_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    respuesta = response.choices[0].message.content

    print("\nRESPUESTA\n")
    print(respuesta)

    archivo = guardar_respuesta(
        tipo=tipo,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        respuesta=respuesta,
        response=response,
    )

    print(f"\nRespuesta guardada en:\n{archivo}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nERROR")
        print("=====")
        print(exc)
        raise SystemExit(1)
