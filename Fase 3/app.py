import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# =========================================================
# RUTAS DEL PROYECTO
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

PROMPTS_DIR = BASE_DIR / "prompts"
CONTEXT_DIR = BASE_DIR / "context"
RESPONSES_DIR = BASE_DIR / "responses"


# =========================================================
# VARIABLES DE ENTORNO
# =========================================================

load_dotenv(BASE_DIR / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")

HF_MODEL = os.getenv(
    "HF_MODEL",
    "meta-llama/Llama-3.1-8B-Instruct",
)

HF_PROVIDER = os.getenv(
    "HF_PROVIDER",
    "auto",
)

MAX_TOKENS = int(
    os.getenv("MAX_TOKENS", "1024")
)

TEMPERATURE = float(
    os.getenv("TEMPERATURE", "0.2")
)


# =========================================================
# ESTRUCTURAS
# =========================================================

@dataclass
class PromptSettings:
    """Contiene los prompts utilizados para construir la conversación.

    Attributes:
        system_prompt: Instrucciones generales del sistema.
        role_prompt: Rol que debe asumir el modelo.
        negative_example: Ejemplo de comportamiento no deseado.
        negative_output: Respuesta considerada incorrecta.
        negative_reasoning: Explicación de por qué la respuesta es incorrecta.
        positive_example: Ejemplo de comportamiento esperado.
        positive_output: Respuesta considerada correcta.
        instruction_prompt: Instrucción final entregada al modelo.
    """

    system_prompt: str
    role_prompt: str

    negative_example: str
    negative_output: str
    negative_reasoning: str

    positive_example: str
    positive_output: str

    instruction_prompt: str


# =========================================================
# LECTURA DE ARCHIVOS
# =========================================================

def read_txt(path: Path) -> str:
    """Lee el contenido de un archivo de texto.

    Args:
        path: Ruta del archivo que se desea leer.

    Returns:
        Contenido del archivo sin espacios al inicio ni al final.

    Raises:
        FileNotFoundError: Si el archivo especificado no existe.
        ValueError: Si el archivo existe pero está vacío.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {path}"
        )

    content = path.read_text(
        encoding="utf-8"
    ).strip()

    if not content:
        raise ValueError(
            f"El archivo está vacío: {path}"
        )

    return content


def load_prompt_settings() -> PromptSettings:
    """Carga todos los prompts comunes utilizados por EcoMarket.

    Los nombres de archivo corresponden exactamente a los archivos
    existentes dentro de la carpeta ``prompts``.

    Returns:
        Instancia de PromptSettings con todos los prompts cargados.

    Raises:
        FileNotFoundError: Si alguno de los archivos requeridos no existe.
        ValueError: Si alguno de los archivos requeridos está vacío.
    """

    return PromptSettings(

        system_prompt=read_txt(
            PROMPTS_DIR / "system.txt"
        ),

        role_prompt=read_txt(
            PROMPTS_DIR / "role_prompt.txt"
        ),

        negative_example=read_txt(
            PROMPTS_DIR / "negative_example.txt"
        ),

        negative_output=read_txt(
            PROMPTS_DIR / "negative_output.txt"
        ),

        negative_reasoning=read_txt(
            PROMPTS_DIR / "negative_reasoning.txt"
        ),

        positive_example=read_txt(
            PROMPTS_DIR / "positive_example.txt"
        ),

        positive_output=read_txt(
            PROMPTS_DIR / "positive_output.txt"
        ),

        instruction_prompt=read_txt(
            PROMPTS_DIR / "instruction_prompt.txt"
        ),
    )


# =========================================================
# SELECCIÓN DEL EJERCICIO
# =========================================================

def get_exercise_files(
    exercise_type: str,
) -> tuple[Path, Path]:
    """Obtiene los archivos correspondientes al ejercicio seleccionado.

    Para ``pedido`` utiliza:
        - prompts/pedido.txt
        - context/pedidos.txt

    Para ``devolucion`` utiliza:
        - prompts/devolucion.txt
        - context/politica_devoluciones.txt

    Args:
        exercise_type: Tipo de ejercicio. Los valores permitidos son
            ``pedido`` y ``devolucion``.

    Returns:
        Tupla que contiene la ruta del prompt específico y la ruta
        del contexto correspondiente.

    Raises:
        ValueError: Si el tipo de ejercicio no es válido.
    """

    exercises = {

        "pedido": {
            "prompt": PROMPTS_DIR / "pedido.txt",
            "context": CONTEXT_DIR / "pedidos.txt",
        },

        "devolucion": {
            "prompt": PROMPTS_DIR / "devolucion.txt",
            "context": (
                CONTEXT_DIR
                / "politica_devoluciones.txt"
            ),
        },
    }

    if exercise_type not in exercises:
        raise ValueError(
            "Tipo de ejercicio inválido. "
            "Utiliza 'pedido' o 'devolucion'."
        )

    selected = exercises[exercise_type]

    return (
        selected["prompt"],
        selected["context"],
    )


def build_content(exercise_type: str) -> str:
    """Construye el caso real que será analizado por el modelo.

    Combina el prompt específico del ejercicio con su correspondiente
    fuente de contexto.

    Args:
        exercise_type: Tipo de ejercicio que se desea ejecutar.
            Puede ser ``pedido`` o ``devolucion``.

    Returns:
        Texto completo del caso real que se enviará dentro de los
        mensajes de la solicitud al modelo.

    Raises:
        ValueError: Si el tipo de ejercicio no es válido.
        FileNotFoundError: Si los archivos necesarios no existen.
    """

    prompt_path, context_path = get_exercise_files(
        exercise_type
    )

    exercise_prompt = read_txt(
        prompt_path
    )

    context = read_txt(
        context_path
    )

    if exercise_type == "pedido":

        context_title = (
            "BASE DE DATOS DE PEDIDOS DE ECOMARKET"
        )

    else:

        context_title = (
            "POLÍTICA DE DEVOLUCIONES DE ECOMARKET"
        )

    content = f"""
SOLICITUD
=========

{exercise_prompt}


{context_title}
{'=' * len(context_title)}

{context}
"""

    return content.strip()


# =========================================================
# CONSTRUCCIÓN DE LOS MENSAJES
# =========================================================

def build_messages(
    settings: PromptSettings,
    content: str,
) -> list[dict[str, str]]:
    """Construye la secuencia de mensajes enviada a Llama.

    La conversación utiliza una combinación de instrucciones del
    sistema, definición de rol, ejemplo negativo, ejemplo positivo
    y el caso real de EcoMarket.

    El ejemplo negativo se incluye como contenido explicativo y no
    como una respuesta del rol ``assistant`` para evitar presentar
    una respuesta incorrecta como comportamiento esperado.

    Args:
        settings: Prompts previamente cargados desde los archivos TXT.
        content: Caso real que el modelo debe resolver.

    Returns:
        Lista de mensajes compatible con Chat Completions.
    """

    messages = [

        # -------------------------------------------------
        # 1. INSTRUCCIÓN GENERAL DEL SISTEMA
        # -------------------------------------------------

        {
            "role": "system",
            "content": settings.system_prompt,
        },


        # -------------------------------------------------
        # 2. DEFINICIÓN DEL ROL
        # -------------------------------------------------

        {
            "role": "system",
            "content": settings.role_prompt,
        },


        # -------------------------------------------------
        # 3. EJEMPLO NEGATIVO
        # -------------------------------------------------

        {
            "role": "user",
            "content": f"""
EJEMPLO NEGATIVO
================

El siguiente ejemplo muestra una respuesta que
NO debe ser reproducida.

CASO
----

{settings.negative_example}


RESPUESTA INCORRECTA
--------------------

{settings.negative_output}


POR QUÉ ES INCORRECTA
---------------------

{settings.negative_reasoning}


IMPORTANTE
----------

Utiliza este ejemplo únicamente para reconocer
los comportamientos que debes evitar.

No reproduzcas la respuesta incorrecta.
""".strip(),
        },


        # -------------------------------------------------
        # 4. EJEMPLO POSITIVO - FEW SHOT
        # -------------------------------------------------

        {
            "role": "user",
            "content": f"""
EJEMPLO POSITIVO
================

{settings.positive_example}
""".strip(),
        },

        {
            "role": "assistant",
            "content": settings.positive_output,
        },


        # -------------------------------------------------
        # 5. CASO REAL
        # -------------------------------------------------

        {
            "role": "user",
            "content": f"""
CASO REAL A RESOLVER
====================

>>>>>

{content}

<<<<<
""".strip(),
        },


        # -------------------------------------------------
        # 6. INSTRUCCIÓN FINAL
        # -------------------------------------------------

        {
            "role": "user",
            "content": settings.instruction_prompt,
        },
    ]

    return messages


# =========================================================
# CONSUMO DE TOKENS
# =========================================================

def get_usage_value(
    usage,
    attribute: str,
    default="No disponible",
):
    """Obtiene un valor del objeto de consumo de tokens.

    Algunos proveedores pueden no retornar todos los campos de uso.
    Esta función evita que la aplicación falle en esos casos.

    Args:
        usage: Objeto de uso retornado por Hugging Face.
        attribute: Nombre del atributo que se desea recuperar.
        default: Valor utilizado cuando el atributo no está disponible.

    Returns:
        Valor correspondiente al atributo solicitado o el valor
        predeterminado si no está disponible.
    """

    if usage is None:
        return default

    try:

        value = getattr(
            usage,
            attribute,
            default,
        )

        if value is None:
            return default

        return value

    except Exception:
        return default


# =========================================================
# GUARDADO DE LA RESPUESTA
# =========================================================

def save_response(
    exercise_type: str,
    response_text: str,
    response,
) -> Path:
    """Guarda la respuesta del modelo en un archivo TXT.

    El nombre del archivo incluye el tipo de ejercicio junto con
    la fecha, hora, minuto y segundo de ejecución.

    También almacena información sobre el modelo y el consumo de
    tokens cuando el proveedor devuelve esos datos.

    Args:
        exercise_type: Tipo de ejercicio ejecutado.
        response_text: Texto generado por el modelo.
        response: Objeto completo retornado por Hugging Face.

    Returns:
        Ruta del archivo TXT generado.
    """

    RESPONSES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    now = datetime.now()

    timestamp = now.strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    output_file = (
        RESPONSES_DIR
        / f"{exercise_type}_{timestamp}.txt"
    )

    usage = getattr(
        response,
        "usage",
        None,
    )

    prompt_tokens = get_usage_value(
        usage,
        "prompt_tokens",
    )

    completion_tokens = get_usage_value(
        usage,
        "completion_tokens",
    )

    total_tokens = get_usage_value(
        usage,
        "total_tokens",
    )

    file_content = f"""
FECHA Y HORA
============
{now.strftime("%Y-%m-%d %H:%M:%S")}


TIPO DE EJERCICIO
=================
{exercise_type}


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


RESPUESTA DEL MODELO
====================

{response_text}
"""

    output_file.write_text(
        file_content.strip(),
        encoding="utf-8",
    )

    return output_file


# =========================================================
# VALIDACIONES
# =========================================================

def validate_environment() -> None:
    """Valida las variables necesarias para ejecutar la aplicación.

    Raises:
        ValueError: Si no se encuentra el token de Hugging Face en
            el archivo ``.env``.
    """

    if not HF_TOKEN:
        raise ValueError(
            "No se encontró HF_TOKEN en el archivo .env."
        )


def get_exercise_from_arguments() -> str:
    """Obtiene el tipo de ejercicio desde los argumentos de consola.

    La aplicación espera uno de los siguientes comandos:

    ``python app.py pedido``

    ``python app.py devolucion``

    Returns:
        Tipo de ejercicio solicitado.

    Raises:
        ValueError: Si no se proporciona un ejercicio válido.
    """

    if len(sys.argv) != 2:

        raise ValueError(
            "Debes indicar qué ejercicio ejecutar.\n\n"
            "Ejemplos:\n"
            "python app.py pedido\n"
            "python app.py devolucion"
        )

    exercise_type = (
        sys.argv[1]
        .strip()
        .lower()
    )

    if exercise_type not in {
        "pedido",
        "devolucion",
    }:

        raise ValueError(
            "El ejercicio debe ser "
            "'pedido' o 'devolucion'."
        )

    return exercise_type


# =========================================================
# EJECUCIÓN PRINCIPAL
# =========================================================

def main() -> None:
    """Ejecuta el flujo principal de EcoMarket.

    El proceso realiza los siguientes pasos:

    1. Valida la configuración del entorno.
    2. Identifica el ejercicio solicitado.
    3. Carga los prompts desde archivos TXT.
    4. Carga el contexto correspondiente.
    5. Construye los mensajes para Llama.
    6. Realiza una única solicitud a Hugging Face.
    7. Muestra la respuesta.
    8. Guarda el resultado en un archivo TXT.

    Raises:
        ValueError: Si existe un problema con la configuración o los
            argumentos de ejecución.
        FileNotFoundError: Si falta alguno de los archivos necesarios.
    """

    # -----------------------------------------------------
    # Validaciones
    # -----------------------------------------------------

    validate_environment()

    exercise_type = get_exercise_from_arguments()


    # -----------------------------------------------------
    # Cargar prompts
    # -----------------------------------------------------

    settings = load_prompt_settings()


    # -----------------------------------------------------
    # Construir caso
    # -----------------------------------------------------

    content = build_content(
        exercise_type
    )


    # -----------------------------------------------------
    # Construir mensajes
    # -----------------------------------------------------

    messages = build_messages(
        settings=settings,
        content=content,
    )


    # -----------------------------------------------------
    # Cliente de Hugging Face
    # -----------------------------------------------------

    client = InferenceClient(
        provider=HF_PROVIDER,
        api_key=HF_TOKEN,
    )


    # -----------------------------------------------------
    # Información de ejecución
    # -----------------------------------------------------

    print("=" * 70)
    print("ECOMARKET - LLAMA 3.1 8B INSTRUCT")
    print("=" * 70)

    print(
        f"Ejercicio  : {exercise_type}"
    )

    print(
        f"Modelo     : {HF_MODEL}"
    )

    print(
        f"Proveedor  : {HF_PROVIDER}"
    )

    print(
        f"Mensajes   : {len(messages)}"
    )

    print("=" * 70)


    # =====================================================
    # UNA ÚNICA SOLICITUD A LLAMA
    # =====================================================

    response = client.chat.completions.create(

        model=HF_MODEL,

        messages=messages,

        temperature=TEMPERATURE,

        max_tokens=MAX_TOKENS,
    )


    # -----------------------------------------------------
    # Obtener respuesta
    # -----------------------------------------------------

    response_text = (
        response
        .choices[0]
        .message
        .content
    )


    # -----------------------------------------------------
    # Mostrar respuesta
    # -----------------------------------------------------

    print("\nRESPUESTA DEL MODELO")
    print("=" * 70)

    print(response_text)


    # -----------------------------------------------------
    # Guardar respuesta
    # -----------------------------------------------------

    output_file = save_response(
        exercise_type=exercise_type,
        response_text=response_text,
        response=response,
    )


    # -----------------------------------------------------
    # Consumo
    # -----------------------------------------------------

    usage = getattr(
        response,
        "usage",
        None,
    )

    print("\nCONSUMO DE TOKENS")
    print("=" * 70)

    print(
        "Entrada :",
        get_usage_value(
            usage,
            "prompt_tokens",
        ),
    )

    print(
        "Salida  :",
        get_usage_value(
            usage,
            "completion_tokens",
        ),
    )

    print(
        "Total   :",
        get_usage_value(
            usage,
            "total_tokens",
        ),
    )


    # -----------------------------------------------------
    # Archivo generado
    # -----------------------------------------------------

    print("\nRESPUESTA GUARDADA EN")
    print("=" * 70)

    print(output_file)


# =========================================================
# PUNTO DE ENTRADA
# =========================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print("\nERROR")
        print("=" * 70)

        print(error)

        raise SystemExit(1)