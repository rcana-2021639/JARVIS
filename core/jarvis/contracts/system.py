"""Contratos del estado global del sistema y su telemetria."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from jarvis.contracts.base import BaseContract, CorrelationId, utc_now
from jarvis.contracts.enums import CascadeLevel, CoreState, PrivacyMode

__all__ = ["StateChange", "SystemState", "Telemetry"]


class Telemetry(BaseContract):
    """Foto instantanea de la salud de la maquina y del propio Jarvis.

    Alimenta el panel de telemetria del HUD (Fase 5), que exige datos reales y
    no simulados. Los campos opcionales son los que no toda maquina expone: en
    una maquina sin GPU dedicada, `vram_used_mb` y `gpu_percent` son None y el
    HUD debe ocultar esos medidores en vez de dibujar ceros.
    """

    cpu_percent: float = Field(ge=0.0, le=100.0, description="Uso total de CPU.")
    cpu_per_core: list[float] = Field(
        default_factory=list,
        description="Uso por nucleo. El HUD lo dibuja como ecualizador vertical.",
    )
    ram_used_mb: float = Field(ge=0.0, description="RAM en uso, en megabytes.")
    ram_total_mb: float = Field(gt=0.0, description="RAM total de la maquina.")
    vram_used_mb: float | None = Field(
        default=None,
        ge=0.0,
        description="VRAM en uso. None si no hay GPU dedicada.",
    )
    vram_total_mb: float | None = Field(
        default=None,
        ge=0.0,
        description="VRAM total. None si no hay GPU dedicada.",
    )
    gpu_percent: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Uso de GPU. None si no se puede medir.",
    )
    disk_free_gb: float | None = Field(
        default=None,
        ge=0.0,
        description="Espacio libre en el disco principal.",
    )
    temperature_c: float | None = Field(
        default=None,
        description="Temperatura del paquete de CPU, si el sistema la expone.",
    )
    net_up_kbps: float = Field(default=0.0, ge=0.0, description="Subida actual.")
    net_down_kbps: float = Field(default=0.0, ge=0.0, description="Bajada actual.")
    captured_at: datetime = Field(
        default_factory=utc_now,
        description="Cuando se tomo la medicion.",
    )

    @property
    def ram_percent(self) -> float:
        """Porcentaje de RAM en uso."""
        return self.ram_used_mb / self.ram_total_mb * 100.0

    @property
    def has_dedicated_gpu(self) -> bool:
        """Si esta maquina tiene GPU dedicada con VRAM medible."""
        return self.vram_total_mb is not None and self.vram_total_mb > 0


class SystemState(BaseContract):
    """El estado global de Jarvis en un instante.

    Es lo que el HUD y la app movil reciben para saber que dibujar: el estado
    del nucleo gobierna la esfera 3D, el modo de privacidad se muestra siempre
    visible, y el gasto del dia alimenta la barra de presupuesto.

    Es un contrato inmutable como todos los demas: cada cambio de estado produce
    una instancia nueva y un evento CORE_STATE_CHANGED. Nadie muta el estado
    global por debajo -- asi el timeline reconstruye la historia completa.
    """

    core_state: CoreState = Field(
        default=CoreState.DORMIDO,
        description="Estado del nucleo. Gobierna la esfera del HUD.",
    )
    privacy_mode: PrivacyMode = Field(
        default=PrivacyMode.HYBRID,
        description="Modo de privacidad activo. LOCAL_ONLY corta el nivel 3.",
    )
    session_id: str | None = Field(
        default=None,
        description="Sesion de conversacion activa. None cuando no hay ninguna abierta.",
    )
    correlation_id: CorrelationId | None = Field(
        default=None,
        description="Interaccion en curso, si la hay.",
    )
    telemetry: Telemetry | None = Field(
        default=None,
        description="Ultima telemetria conocida. None mientras el nucleo arranca.",
    )
    active_level: CascadeLevel | None = Field(
        default=None,
        description="Nivel de la cascada que atiende ahora mismo. None si esta ocioso.",
    )
    spent_today_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Gasto acumulado del dia en la API de la nube.",
    )
    budget_today_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Presupuesto diario configurado. Cero significa sin limite.",
    )
    microphone_muted: bool = Field(
        default=False,
        description="Si el microfono esta silenciado por el usuario o por el killswitch.",
    )
    updated_at: datetime = Field(
        default_factory=utc_now,
        description="Cuando se produjo este estado.",
    )

    @property
    def is_busy(self) -> bool:
        """Si el nucleo esta ocupado y no puede atender una orden nueva de inmediato."""
        return self.core_state in {
            CoreState.PENSANDO,
            CoreState.EJECUTANDO,
            CoreState.CONFIRMANDO,
        }

    @property
    def budget_exhausted(self) -> bool:
        """Si se agoto el presupuesto del dia.

        Al agotarse, la Fase 2 avisa por voz y degrada a modo solo-local en vez
        de seguir cobrando. Un presupuesto de cero significa sin limite.
        """
        return self.budget_today_usd > 0 and self.spent_today_usd >= self.budget_today_usd

    @property
    def cloud_allowed(self) -> bool:
        """Si en este momento esta permitido llegar al nivel 3 de la cascada."""
        return self.privacy_mode is not PrivacyMode.LOCAL_ONLY and not self.budget_exhausted


class StateChange(BaseContract):
    """Una transicion del estado del nucleo. Viaja en `core.state.changed`.

    Lleva el estado anterior y el nuevo porque el timeline del HUD dibuja
    transiciones, no estados sueltos: "ESCUCHANDO -> HABLANDO por barge-in" dice
    mucho mas que "HABLANDO".
    """

    previous: CoreState = Field(description="Estado del que se sale.")
    current: CoreState = Field(description="Estado al que se entra.")
    reason: str = Field(
        min_length=1,
        description="Causa corta y legible: wake, silencio, barge_in, respuesta...",
    )
    source: str = Field(min_length=1, description="Modulo que provoco la transicion.")
    correlation_id: CorrelationId | None = Field(default=None)
    changed_at: datetime = Field(default_factory=utc_now)
