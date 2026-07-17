"""
Chat por consola con el Agente de Soporte de BimBam Buy.

Ejecutar con:
    python main_cli.py
"""
from src.agent import AgenteBimBamBuy

BANNER = """\
╔══════════════════════════════════════════════════╗
║   🤖  Agente de Soporte — BimBam Buy               ║
║   Escribe tu pregunta o 'salir' para terminar.     ║
║   Escribe 'reiniciar' para borrar el historial.    ║
╚══════════════════════════════════════════════════╝
"""


def main() -> None:
    print(BANNER)
    print("⏳ Cargando agente (puede tardar unos segundos la primera vez)...\n")
    agente = AgenteBimBamBuy()
    print("✅ ¡Listo! ¿En qué te puedo ayudar hoy?\n")

    while True:
        try:
            pregunta = input("Tú: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 ¡Hasta luego!")
            break

        if not pregunta:
            continue
        if pregunta.lower() in {"salir", "exit", "quit"}:
            print("👋 ¡Hasta luego!")
            break
        if pregunta.lower() == "reiniciar":
            agente.reiniciar_conversacion()
            print("🔄 Historial reiniciado.\n")
            continue

        respuesta = agente.responder(pregunta)
        print(f"\nAgente: {respuesta.contenido}")
        if respuesta.fuentes:
            print(f"📎 Fuentes consultadas: {', '.join(respuesta.fuentes)}")
        print()


if __name__ == "__main__":
    main()
