"""Nucleo del Proyecto JARVIS.

Un asistente personal residente controlado por voz, que corre como servicio
permanente, opera el sistema operativo, recuerda todo el historial y se presenta
mediante un HUD tridimensional reactivo al audio.

Estructura del paquete:

    contracts/   los contratos de datos -- el idioma comun de todo lo demas
    bus/         publicacion y suscripcion de eventos entre modulos
    config.py    configuracion con jerarquia y recarga en caliente
    logging.py   registro estructurado y trazado de duraciones

    voice/       pipeline de audio                    (Fase 1)
    cognition/   la cascada de tres niveles           (Fase 2)
    memory/      las cinco capas de memoria           (Fase 3)
    tools/       servidores MCP                       (Fases 6 y 7)
    guardian/    permisos, boveda y auditoria         (Fase 8)
    bridge/      puente remoto cifrado                (Fase 9)
    service/     daemon, supervisor y bandeja         (Fase 11)

Los paquetes vacios son marcadores de las fases que vienen. Que existan desde el
principio no es adorno: fija el sitio de cada cosa antes de que haya presion por
meterla donde sea.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
