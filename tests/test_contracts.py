"""Pruebas de los contratos de datos.

Lo que se verifica aqui no es que Pydantic funcione -- eso ya esta probado por
Pydantic. Se verifica que las REGLAS DE NEGOCIO del documento maestro esten
codificadas: que un ToolResult fallido no pueda existir sin explicacion, que una
accion ROJA no pueda registrarse como ejecutada sin confirmacion, que un nivel
sin costo no pueda reportar gasto.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from jarvis.contracts import (
    AuditEntry,
    CascadeLevel,
    Command,
    CommandOrigin,
    CoreState,
    Emotion,
    Event,
    EventType,
    Intent,
    Language,
    MemoryKind,
    MemoryRecord,
    PrivacyMode,
    Response,
    RiskLevel,
    SystemState,
    Telemetry,
    ToolCall,
    ToolResult,
    Utterance,
    new_correlation_id,
    utc_now,
)

# ---------------------------------------------------------------------------
# Validacion basica
# ---------------------------------------------------------------------------


def test_utterance_rechaza_confianza_fuera_de_rango(correlation_id: str) -> None:
    """La confianza es una probabilidad: fuera de [0,1] es un dato corrupto."""
    with pytest.raises(ValidationError):
        Utterance(
            text="hola",
            confidence=1.5,
            duration_ms=100.0,
            correlation_id=correlation_id,
        )


def test_utterance_rechaza_fin_anterior_al_inicio(correlation_id: str) -> None:
    """El audio no puede terminar antes de empezar."""
    ahora = utc_now()
    with pytest.raises(ValidationError, match="anterior a"):
        Utterance(
            text="hola",
            confidence=0.9,
            duration_ms=100.0,
            started_at=ahora,
            ended_at=ahora - timedelta(seconds=5),
            correlation_id=correlation_id,
        )


def test_utterance_detecta_transcripcion_vacia(correlation_id: str) -> None:
    """Decir la palabra clave y quedarse callado es un caso normal, no un error."""
    vacia = Utterance(
        text="   ",
        confidence=0.1,
        duration_ms=50.0,
        correlation_id=correlation_id,
    )
    assert vacia.is_empty
    assert not Utterance(
        text="algo",
        confidence=0.9,
        duration_ms=50.0,
        correlation_id=correlation_id,
    ).is_empty


def test_contratos_rechazan_campos_desconocidos(correlation_id: str) -> None:
    """Un campo de mas es un error, no algo que se ignora en silencio."""
    with pytest.raises(ValidationError):
        Intent(
            name="chat.free",
            confidence=0.5,
            suggested_level=CascadeLevel.LOCAL_LLM,
            nivel_sugerido=2,  # type: ignore[call-arg]
        )


def test_contratos_son_inmutables(utterance: Utterance) -> None:
    """Un mensaje ya publicado no puede modificarse por debajo a nadie."""
    with pytest.raises(ValidationError):
        utterance.text = "otra cosa"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Reglas de negocio
# ---------------------------------------------------------------------------


def test_tool_result_fallido_exige_explicacion() -> None:
    """Nada falla en silencio: un fallo sin motivo no es un resultado valido."""
    with pytest.raises(ValidationError, match="debe traer 'error'"):
        ToolResult(call_id="abc", success=False, duration_ms=12.0)


def test_tool_result_exitoso_no_puede_traer_error() -> None:
    """Exito y error a la vez es incoherente y esconde bugs."""
    with pytest.raises(ValidationError, match="no puede traer 'error'"):
        ToolResult(call_id="abc", success=True, error="algo", duration_ms=12.0)


def test_tool_result_valido_en_ambos_sentidos() -> None:
    """Los dos casos coherentes se construyen sin problema."""
    ok = ToolResult(call_id="a", success=True, data={"volumen": 70}, duration_ms=5.0)
    fallo = ToolResult(call_id="b", success=False, error="dispositivo ausente", duration_ms=3.0)
    assert ok.data == {"volumen": 70}
    assert fallo.error


def test_response_de_nivel_sin_costo_no_puede_cobrar(correlation_id: str) -> None:
    """Cobrar por algo que nunca salio a la nube es un error de contabilidad."""
    with pytest.raises(ValidationError, match="no tiene costo"):
        Response(
            text="Listo.",
            handled_by_level=CascadeLevel.FAST_PATH,
            estimated_cost_usd=0.01,
            correlation_id=correlation_id,
        )


def test_response_de_nivel_nube_si_puede_cobrar(correlation_id: str) -> None:
    """El nivel 3 es el unico que tiene costo."""
    respuesta = Response(
        text="Analice el archivo y encontre tres problemas.",
        handled_by_level=CascadeLevel.CLOUD_LLM,
        estimated_cost_usd=0.0042,
        correlation_id=correlation_id,
    )
    assert respuesta.estimated_cost_usd > 0


def test_response_detecta_respuestas_demasiado_largas_para_hablar(correlation_id: str) -> None:
    """Un ensayo se lee en el chat, no se dicta a los oidos."""
    corta = Response(
        text="Listo. Volumen al setenta.",
        handled_by_level=CascadeLevel.FAST_PATH,
        correlation_id=correlation_id,
    )
    larga = Response(
        text="Primero. Segundo. Tercero. Cuarto.",
        handled_by_level=CascadeLevel.FAST_PATH,
        correlation_id=correlation_id,
    )
    assert corta.is_spoken_length_ok
    assert not larga.is_spoken_length_ok
    # Si no se va a hablar, la longitud da igual.
    assert larga.model_copy(update={"should_speak": False}).is_spoken_length_ok


def test_auditoria_rechaza_accion_roja_ejecutada_sin_confirmacion(correlation_id: str) -> None:
    """Una bitacora que registra eso estaria mintiendo sobre lo que paso."""
    # Se busca solo "confirmacion" y no la frase completa: Pydantic parte los
    # mensajes largos en varias lineas y el salto caeria en medio de la frase.
    with pytest.raises(ValidationError, match=r"confirmacion"):
        AuditEntry(
            action="files.delete",
            level=RiskLevel.ROJO,
            origin=CommandOrigin.VOZ,
            confirmed=False,
            success=True,
            correlation_id=correlation_id,
        )


def test_auditoria_acepta_accion_roja_confirmada(correlation_id: str) -> None:
    """Con confirmacion hablada si se puede registrar."""
    entrada = AuditEntry(
        action="files.delete",
        level=RiskLevel.ROJO,
        origin=CommandOrigin.VOZ,
        confirmed=True,
        success=True,
        correlation_id=correlation_id,
    )
    assert entrada.confirmed


def test_auditoria_rechaza_accion_negra_exitosa(correlation_id: str) -> None:
    """El nivel NEGRO se rechaza siempre: no hay frase que lo autorice."""
    with pytest.raises(ValidationError, match="NEGRO"):
        AuditEntry(
            action="guardian.disable",
            level=RiskLevel.NEGRO,
            origin=CommandOrigin.VOZ,
            confirmed=True,
            success=True,
            correlation_id=correlation_id,
        )


def test_auditoria_redacta_credenciales() -> None:
    """El valor de una credencial nunca aparece; solo el marcador."""
    assert AuditEntry.redact_credential("github") == "<credencial:github>"


# ---------------------------------------------------------------------------
# Memoria
# ---------------------------------------------------------------------------


def test_memoria_permanente_no_decae() -> None:
    """Lo marcado como permanente sobrevive a la consolidacion nocturna."""
    permanente = MemoryRecord(
        kind=MemoryKind.SEMANTICA,
        content="El usuario se llama Rhandy",
        importance=1.0,
        last_accessed_at=datetime(2020, 1, 1, tzinfo=UTC),
    )
    assert permanente.is_permanent
    assert permanente.decayed_at(utc_now(), half_life_days=30.0) == 0.0
    assert permanente.effective_weight == 1.0


def test_memoria_decae_una_mitad_por_vida_media() -> None:
    """Pasada una vida media sin usarse, el recuerdo perdio la mitad del peso."""
    hace_treinta_dias = utc_now() - timedelta(days=30)
    registro = MemoryRecord(
        kind=MemoryKind.SEMANTICA,
        content="prefiere el modo oscuro",
        importance=0.8,
        last_accessed_at=hace_treinta_dias,
    )
    decaimiento = registro.decayed_at(utc_now(), half_life_days=30.0)
    assert decaimiento == pytest.approx(0.5, abs=0.01)


def test_memoria_rechaza_vida_media_invalida() -> None:
    """Una vida media de cero dividiria por cero."""
    registro = MemoryRecord(kind=MemoryKind.EPISODICA, content="algo")
    with pytest.raises(ValueError, match="mayor que cero"):
        registro.decayed_at(utc_now(), half_life_days=0.0)


def test_memoria_peso_efectivo_baja_con_el_decaimiento() -> None:
    """El peso efectivo es lo que ordena los candidatos en la recuperacion."""
    registro = MemoryRecord(
        kind=MemoryKind.SEMANTICA,
        content="algo",
        importance=0.8,
        decay=0.5,
    )
    assert registro.effective_weight == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Estado del sistema
# ---------------------------------------------------------------------------


def test_estado_detecta_cuando_esta_ocupado() -> None:
    """PENSANDO, EJECUTANDO y CONFIRMANDO no pueden atender otra orden."""
    for estado in (CoreState.PENSANDO, CoreState.EJECUTANDO, CoreState.CONFIRMANDO):
        assert SystemState(core_state=estado).is_busy
    for estado in (CoreState.DORMIDO, CoreState.ESCUCHANDO, CoreState.HABLANDO):
        assert not SystemState(core_state=estado).is_busy


def test_modo_local_only_corta_la_nube() -> None:
    """LOCAL_ONLY significa que ni un byte sale de la maquina."""
    assert not SystemState(privacy_mode=PrivacyMode.LOCAL_ONLY).cloud_allowed
    assert SystemState(privacy_mode=PrivacyMode.HYBRID).cloud_allowed


def test_presupuesto_agotado_degrada_a_local() -> None:
    """Al llegar al limite se avisa y se degrada, no se sigue cobrando."""
    agotado = SystemState(spent_today_usd=0.50, budget_today_usd=0.50)
    assert agotado.budget_exhausted
    assert not agotado.cloud_allowed

    con_saldo = SystemState(spent_today_usd=0.10, budget_today_usd=0.50)
    assert not con_saldo.budget_exhausted
    assert con_saldo.cloud_allowed


def test_presupuesto_cero_significa_sin_limite() -> None:
    """Un presupuesto sin configurar no debe bloquear la nube."""
    sin_limite = SystemState(spent_today_usd=99.0, budget_today_usd=0.0)
    assert not sin_limite.budget_exhausted


def test_telemetria_sin_gpu_dedicada() -> None:
    """Esta maquina no tiene GPU dedicada: el HUD debe poder distinguirlo."""
    sin_gpu = Telemetry(cpu_percent=30.0, ram_used_mb=8000.0, ram_total_mb=16000.0)
    assert not sin_gpu.has_dedicated_gpu
    assert sin_gpu.ram_percent == pytest.approx(50.0)

    con_gpu = Telemetry(
        cpu_percent=30.0,
        ram_used_mb=8000.0,
        ram_total_mb=16000.0,
        vram_used_mb=2000.0,
        vram_total_mb=12000.0,
    )
    assert con_gpu.has_dedicated_gpu


# ---------------------------------------------------------------------------
# Serializacion de ida y vuelta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("campo", ["intent", "raw_text", "origin", "correlation_id"])
def test_command_sobrevive_la_ida_y_vuelta(command: Command, campo: str) -> None:
    """Serializar y deserializar no puede perder ni alterar nada."""
    recuperado = Command.from_wire(command.to_wire())
    assert getattr(recuperado, campo) == getattr(command, campo)


def test_ida_y_vuelta_conserva_los_enums_como_enums(command: Command) -> None:
    """Al volver del cable, un enum debe ser un enum otra vez y no un string."""
    recuperado = Command.from_wire(command.to_wire())
    assert recuperado.origin is CommandOrigin.VOZ
    assert recuperado.intent.suggested_level is CascadeLevel.FAST_PATH


def test_serializacion_produce_tipos_json(utterance: Utterance) -> None:
    """Lo que sale al cable tiene que ser JSON puro: el HUD es TypeScript."""
    cable = utterance.to_wire()
    assert isinstance(cable["language"], str)
    assert isinstance(cable["started_at"], str)
    assert cable["language"] == Language.ES.value


def test_response_con_acciones_sobrevive_la_ida_y_vuelta(
    tool_call: ToolCall,
    correlation_id: str,
) -> None:
    """Los contratos anidados tambien deben viajar completos."""
    original = Response(
        text="Listo.",
        emotion=Emotion.AFIRMATIVA,
        handled_by_level=CascadeLevel.FAST_PATH,
        actions=[tool_call],
        correlation_id=correlation_id,
    )
    recuperado = Response.from_wire(original.to_wire())
    assert len(recuperado.actions) == 1
    assert recuperado.actions[0].qualified_name == "media.volume_up"
    assert recuperado.actions[0].risk_level is RiskLevel.VERDE


# ---------------------------------------------------------------------------
# Eventos
# ---------------------------------------------------------------------------


def test_evento_envuelve_y_recupera_un_contrato(utterance: Utterance) -> None:
    """`Event.of` y `payload_as` son la forma tipada de usar el bus."""
    evento = Event.of(
        EventType.VOICE_STT_FINAL,
        utterance,
        source="voice.stt",
        correlation_id=utterance.correlation_id,
    )
    recuperada = evento.payload_as(Utterance)
    assert recuperada.text == utterance.text
    assert recuperada.is_final == utterance.is_final


def test_payload_as_falla_si_la_carga_no_encaja(response: Response) -> None:
    """Un tipo de evento con carga de otra forma debe explotar, no corromper."""
    evento = Event.of(EventType.COGNITION_RESPONSE, response, source="router")
    with pytest.raises(ValidationError):
        evento.payload_as(Utterance)


@pytest.mark.parametrize(
    ("patron", "esperado"),
    [
        ("voice.*", True),
        ("voice.stt.*", True),
        ("*", True),
        ("*.final", True),
        ("cognition.*", False),
        ("voice.tts.*", False),
    ],
)
def test_coincidencia_por_patron_glob(utterance: Utterance, patron: str, esperado: bool) -> None:
    """El reparto por glob es lo que permite que el HUD escuche familias enteras."""
    evento = Event.of(EventType.VOICE_STT_FINAL, utterance, source="voice.stt")
    assert evento.matches(patron) is esperado


def test_evento_calcula_su_propia_edad() -> None:
    """El bus mide latencia con esto."""
    viejo = Event(
        type=EventType.CORE_STARTED,
        source="core",
        timestamp=utc_now() - timedelta(milliseconds=500),
    )
    assert viejo.age_ms >= 500


def test_todos_los_estados_del_nucleo_estan_definidos() -> None:
    """Los ocho estados visuales del documento maestro, ni uno menos."""
    assert len(CoreState) == 8
    assert {estado.value for estado in CoreState} == {
        "DORMIDO",
        "DESPIERTO",
        "ESCUCHANDO",
        "PENSANDO",
        "HABLANDO",
        "EJECUTANDO",
        "ERROR",
        "CONFIRMANDO",
    }


def test_los_cuatro_niveles_de_riesgo_estan_definidos() -> None:
    """VERDE, AMBAR, ROJO y NEGRO, con valores sin acentos para TOML y JSON."""
    assert {nivel.value for nivel in RiskLevel} == {"VERDE", "AMBAR", "ROJO", "NEGRO"}


def test_los_niveles_de_cascada_se_ordenan() -> None:
    """El enrutador compara niveles con < y >, asi que el orden debe ser real."""
    assert CascadeLevel.FAST_PATH < CascadeLevel.LOCAL_LLM < CascadeLevel.CLOUD_LLM


def test_correlacion_es_unica() -> None:
    """Dos interacciones distintas nunca comparten id."""
    assert new_correlation_id() != new_correlation_id()
