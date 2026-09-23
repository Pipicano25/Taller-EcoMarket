import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# RUTAS DEL PROYECTO
BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"
CONTEXT_DIR = BASE_DIR / "context"
REQUESTS_DIR = BASE_DIR / "requests"
RESPONSES_DIR = BASE_DIR / "responses"

# VARIABLES DE ENTORNO
load_dotenv(BASE_DIR / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct",)
HF_PROVIDER = os.getenv("HF_PROVIDER", "auto",)
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))


@dataclass
class PromptSettings:
    """Agrupa los prompts generales utilizados por EcoMarket.

    Attributes:
        system_prompt: Reglas generales que debe seguir el modelo.
        role_prompt: Rol que debe asumir el asistente de EcoMarket.
        negative_example: Ejemplo de una situación mal resuelta.
        negative_output: Ejemplo de una respuesta incorrecta.
        negative_reasoning: Explicación de por qué la respuesta es incorrecta.
        positive_example: Ejemplo de una situación correctamente planteada.
        positive_output: Ejemplo de la respuesta esperada.
        instruction_prompt: Procedimiento general para analizar el caso real.
    """

    system_prompt: str
    role_prompt: str
    negative_example: str
    negative_output: str
    negative_reasoning: str
    positive_example: str
    positive_output: str
    instruction_prompt: str


def read_txt(path: Path) -> str:
    """Lee el contenido de un archivo TXT.

    Args:
        path: Ruta del archivo que se desea leer.

    Returns:
        Contenido del archivo sin espacios adicionales al inicio
        ni al final.

    Raises:
        FileNotFoundError: Si el archivo no existe.
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
    """Carga los prompts generales desde la carpeta ``prompts``.

    Returns:
        Instancia de PromptSettings con todos los archivos de
        configuración cargados.

    Raises:
        FileNotFoundError: Si falta alguno de los archivos requeridos.
        ValueError: Si alguno de los archivos está vacío.
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


# CONFIGURACIÓN DE LOS EJERCICIOS
def get_exercise_files(
    exercise_type: str,
) -> tuple[Path, list[Path], Path]:
    """Obtiene los archivos necesarios para el tipo de solicitud.

    Para una consulta de pedido se utilizan:
        - prompts/pedido.txt
        - context/pedidos.txt
        - requests/pedido.txt

    Para una devolución se utilizan:
        - prompts/devolucion.txt
        - context/pedidos.txt
        - context/politica_devoluciones.txt
        - requests/devolucion.txt

    Args:
        exercise_type: Tipo de solicitud. Los valores soportados son
            ``pedido`` y ``devolucion``.

    Returns:
        Tupla con:
            - Ruta del prompt específico.
            - Lista de rutas de contexto.
            - Ruta de la solicitud real del cliente.

    Raises:
        ValueError: Si el tipo de solicitud no está soportado.
    """

    exercises = {
        "pedido": {
            "prompt": PROMPTS_DIR / "pedido.txt",
            "contexts": [
                CONTEXT_DIR / "pedidos.txt",
            ],
            "request": REQUESTS_DIR / "pedido.txt",
        },

        "devolucion": {
            "prompt": PROMPTS_DIR / "devolucion.txt",
            "contexts": [
                CONTEXT_DIR / "pedidos.txt",
                CONTEXT_DIR / "politica_devoluciones.txt",
            ],
            "request": REQUESTS_DIR / "devolucion.txt",
        },
    }

    if exercise_type not in exercises:
        raise ValueError(
            "Tipo de solicitud inválido. "
            "Utiliza 'pedido' o 'devolucion'."
        )

    selected = exercises[exercise_type]

    return (
        selected["prompt"],
        selected["contexts"],
        selected["request"],
    )


def build_context(
    context_paths: list[Path],
) -> str:
    """Construye el contexto utilizando una o varias fuentes.

    Cada archivo se identifica explícitamente dentro del contenido
    enviado al modelo para que pueda distinguir entre la base de
    pedidos y las políticas de EcoMarket.

    Args:
        context_paths: Lista de archivos que contienen información
            de contexto.

    Returns:
        Texto combinado con todas las fuentes disponibles.

    Raises:
        FileNotFoundError: Si alguno de los archivos no existe.
        ValueError: Si alguno de los archivos está vacío.
    """

    context_sections = []

    for context_path in context_paths:
        context_content = read_txt(
            context_path
        )

        section = f"""
FUENTE: {context_path.name}
{'=' * (8 + len(context_path.name))}

{context_content}
""".strip()

        context_sections.append(
            section
        )

    return "\n\n".join(
        context_sections
    )


def build_content(
    exercise_type: str,
) -> str:
    """Construye el caso real que debe analizar el modelo.

    La información se mantiene separada conceptualmente en tres partes:

    1. Solicitud real del cliente.
    2. Instrucciones específicas para el tipo de solicitud.
    3. Información disponible de EcoMarket.

    Args:
        exercise_type: Tipo de solicitud que será procesada.

    Returns:
        Caso completo que será enviado al modelo.

    Raises:
        FileNotFoundError: Si falta alguno de los archivos requeridos.
        ValueError: Si algún archivo está vacío o el tipo de ejercicio
            no está soportado.
    """

    (
        prompt_path,
        context_paths,
        request_path,
    ) = get_exercise_files(
        exercise_type
    )

    client_request = read_txt(
        request_path
    )

    exercise_prompt = read_txt(
        prompt_path
    )

    combined_context = build_context(
        context_paths
    )

    content = f"""
SOLICITUD REAL DEL CLIENTE
==========================

{client_request}

INSTRUCCIONES PARA ESTE TIPO DE SOLICITUD
=========================================

{exercise_prompt}

INFORMACIÓN DISPONIBLE DE ECOMARKET
===================================

{combined_context}
"""

    return content.strip()


# CONSTRUCCIÓN DE MESSAGES
def build_messages(
    settings: PromptSettings,
    content: str,
) -> list[dict[str, str]]:
    """Construye los mensajes enviados al modelo.

    La estructura combina:

    - Reglas generales.
    - Rol del asistente.
    - Ejemplo negativo.
    - Explicación del error.
    - Ejemplo positivo.
    - Respuesta positiva esperada.
    - Caso real.
    - Instrucción final.

    El ejemplo negativo se incluye como contenido de usuario en lugar
    de utilizar el rol ``assistant`` para evitar presentar una respuesta
    incorrecta como conducta esperada.

    Args:
        settings: Configuración de prompts cargada desde archivos TXT.
        content: Caso real que debe resolver el modelo.

    Returns:
        Lista de mensajes compatible con Chat Completions.
    """

    messages = [
        # REGLAS GENERALES
        {
            "role": "system",
            "content": settings.system_prompt,
        },
        # ROL DEL ASISTENTE
        {
            "role": "system",
            "content": settings.role_prompt,
        },
        # EJEMPLO NEGATIVO
        {
            "role": "user",
            "content": f"""
EJEMPLO NEGATIVO
================

El siguiente caso representa una forma incorrecta
de responder.

CASO
----

{settings.negative_example}


RESPUESTA INCORRECTA
--------------------

{settings.negative_output}


POR QUÉ LA RESPUESTA ES INCORRECTA
----------------------------------

{settings.negative_reasoning}


IMPORTANTE
----------

No reproduzcas la respuesta incorrecta.

Utiliza este ejemplo únicamente para identificar
qué comportamientos, omisiones y conclusiones debes evitar.
""".strip(),
        },

        # EJEMPLO POSITIVO
        {
            "role": "user",
            "content": f"""
EJEMPLO POSITIVO
================

El siguiente caso representa el comportamiento
esperado.

{settings.positive_example}
""".strip(),
        },

        {
            "role": "assistant",
            "content": settings.positive_output,
        },
        # CASO REAL
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
        # INSTRUCCIÓN FINAL
        {
            "role": "user",
            "content": settings.instruction_prompt,
        },
    ]

    return messages


# CONSUMO DE TOKENS
def get_usage_value(
    usage: Any,
    attribute: str,
    default: Any = "No disponible",
) -> Any:
    """Obtiene de forma segura un valor del objeto de uso.

    Algunos proveedores de Hugging Face pueden no retornar todos los
    campos relacionados con el consumo de tokens. Esta función evita
    que la aplicación falle si un atributo no está disponible.

    Args:
        usage: Objeto de uso retornado por el proveedor.
        attribute: Nombre del atributo que se desea recuperar.
        default: Valor utilizado cuando el atributo no está disponible.

    Returns:
        Valor correspondiente al atributo solicitado o ``default``.
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


# GUARDADO DE RESPUESTAS
def save_response(
    exercise_type: str,
    response_text: str,
    response: Any,
) -> Path:
    """Guarda la respuesta del modelo en un archivo TXT.

    El nombre del archivo contiene el tipo de solicitud y una marca
    de tiempo con año, mes, día, hora, minuto y segundo.

    Args:
        exercise_type: Tipo de solicitud procesada.
        response_text: Respuesta generada por Llama.
        response: Objeto completo retornado por Hugging Face.

    Returns:
        Ruta del archivo de respuesta generado.
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


TIPO DE SOLICITUD
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


# VALIDACIÓN
def validate_environment() -> None:
    """Valida la configuración mínima de la aplicación.

    Raises:
        ValueError: Si el token de Hugging Face no está configurado.
    """

    if not HF_TOKEN:
        raise ValueError(
            "No se encontró HF_TOKEN en el archivo .env."
        )


def get_exercise_from_arguments() -> str:
    """Obtiene el tipo de solicitud desde la línea de comandos.

    Ejemplos válidos:

    ``python app.py pedido``

    ``python app.py devolucion``

    Returns:
        Tipo de solicitud solicitado.

    Raises:
        ValueError: Si no se proporciona un argumento válido.
    """

    if len(sys.argv) != 2:
        raise ValueError(
            "Debes indicar el tipo de solicitud.\n\n"
            "Ejemplos:\n"
            "python app.py pedido\n"
            "python app.py devolucion"
        )

    exercise_type = (
        sys.argv[1]
        .strip()
        .lower()
    )

    valid_exercises = {
        "pedido",
        "devolucion",
    }

    if exercise_type not in valid_exercises:
        raise ValueError(
            "El tipo de solicitud debe ser "
            "'pedido' o 'devolucion'."
        )

    return exercise_type


# CLIENTE HUGGING FACE
def create_client() -> InferenceClient:
    """Crea el cliente utilizado para consumir Hugging Face.

    Returns:
        Cliente InferenceClient configurado con el proveedor y token
        definidos en el archivo ``.env``.
    """

    return InferenceClient(
        provider=HF_PROVIDER,
        api_key=HF_TOKEN,
    )


# EJECUCIÓN DE LLAMA
def generate_response(
    client: InferenceClient,
    messages: list[dict[str, str]],
) -> Any:
    """Realiza una única solicitud al modelo Llama.

    Esta función representa el único punto del programa donde se
    realiza una llamada de inferencia a Hugging Face.

    Args:
        client: Cliente autenticado de Hugging Face.
        messages: Lista completa de mensajes que conforman el contexto
            de la conversación.

    Returns:
        Respuesta completa retornada por Hugging Face.
    """

    response = client.chat.completions.create(
        model=HF_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    return response


# INFORMACIÓN DE CONSOLA
def print_execution_info(
    exercise_type: str,
    messages: list[dict[str, str]],
) -> None:
    """Muestra información básica de la ejecución.

    Args:
        exercise_type: Tipo de solicitud seleccionada.
        messages: Lista de mensajes que serán enviados al modelo.
    """

    print("=" * 70)
    print("ECOMARKET - LLAMA 3.1 8B INSTRUCT")
    print("=" * 70)

    print(
        f"Tipo de solicitud : {exercise_type}"
    )

    print(
        f"Modelo            : {HF_MODEL}"
    )

    print(
        f"Proveedor         : {HF_PROVIDER}"
    )

    print(
        f"Mensajes enviados : {len(messages)}"
    )

    print("=" * 70)


def print_usage(
    response: Any,
) -> None:
    """Muestra el consumo de tokens de la solicitud.

    Args:
        response: Respuesta completa retornada por Hugging Face.
    """

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


# MAIN
def main() -> None:
    """Ejecuta el flujo completo de EcoMarket.

    El proceso realiza:

    1. Validación de variables de entorno.
    2. Identificación del tipo de solicitud.
    3. Lectura de los prompts.
    4. Lectura de la solicitud real del cliente.
    5. Lectura de las fuentes de contexto correspondientes.
    6. Construcción del caso real.
    7. Construcción de los mensajes.
    8. Ejecución de una única solicitud a Llama.
    9. Presentación de la respuesta.
    10. Registro del consumo de tokens.
    11. Almacenamiento de la respuesta en un archivo TXT.

    Raises:
        ValueError: Si la configuración o los argumentos son inválidos.
        FileNotFoundError: Si falta alguno de los archivos requeridos.
    """

    # --------------------------------------------------------
    # Validar configuración
    # --------------------------------------------------------

    validate_environment()

    # --------------------------------------------------------
    # Determinar tipo de solicitud
    # --------------------------------------------------------

    exercise_type = get_exercise_from_arguments()

    # --------------------------------------------------------
    # Cargar prompts
    # --------------------------------------------------------

    settings = load_prompt_settings()

    # --------------------------------------------------------
    # Construir caso real
    # --------------------------------------------------------

    content = build_content(
        exercise_type
    )

    # --------------------------------------------------------
    # Construir messages
    # --------------------------------------------------------

    messages = build_messages(
        settings=settings,
        content=content,
    )

    # --------------------------------------------------------
    # Crear cliente
    # --------------------------------------------------------

    client = create_client()

    # --------------------------------------------------------
    # Mostrar información
    # --------------------------------------------------------

    print_execution_info(
        exercise_type=exercise_type,
        messages=messages,
    )

    # ========================================================
    # UNA ÚNICA LLAMADA A LLAMA
    # ========================================================

    response = generate_response(
        client=client,
        messages=messages,
    )

    # --------------------------------------------------------
    # Obtener respuesta
    # --------------------------------------------------------

    response_text = (
        response
        .choices[0]
        .message
        .content
    )

    # --------------------------------------------------------
    # Mostrar respuesta
    # --------------------------------------------------------

    print("\nRESPUESTA DEL MODELO")
    print("=" * 70)

    print(response_text)

    # --------------------------------------------------------
    # Mostrar consumo
    # --------------------------------------------------------

    print_usage(
        response
    )

    # --------------------------------------------------------
    # Guardar respuesta
    # --------------------------------------------------------

    output_file = save_response(
        exercise_type=exercise_type,
        response_text=response_text,
        response=response,
    )

    print("\nRESPUESTA GUARDADA EN")
    print("=" * 70)

    print(output_file)


# PUNTO DE ENTRADA
if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\nERROR")
        print("=" * 70)

        print(error)

        raise SystemExit(1)
