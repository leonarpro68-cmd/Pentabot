"""Lector del mando DualSense (PS5) por evdev, para pilotar el bipedo.

No necesita sudo: udev etiqueta el mando como joystick (regla 70-uaccess.rules)
y da permiso de lectura a la sesion activa.
"""

from __future__ import annotations

import threading

import evdev
from evdev import InputDevice, ecodes

ZONA_MUERTA = 0.12


def _sin_cero_negativo(v: float) -> float:
  """Convierte -0.0 en 0.0, para que los volcados se lean bien."""
  return 0.0 if v == 0.0 else v


def buscar_mando(nombre_parcial: str | None = None) -> InputDevice:
  """Devuelve el primer dispositivo que parezca un mando de juego."""
  candidatos = []
  for ruta in evdev.list_devices():
    try:
      d = InputDevice(ruta)
    except OSError:
      continue
    caps = d.capabilities()
    ejes = {c for c, _ in caps.get(ecodes.EV_ABS, [])}
    botones = set(caps.get(ecodes.EV_KEY, []))
    # Un mando tiene los dos sticks y el boton sur (X en PlayStation).
    if {ecodes.ABS_X, ecodes.ABS_Y, ecodes.ABS_RX}.issubset(ejes) and ecodes.BTN_SOUTH in botones:
      if nombre_parcial and nombre_parcial.lower() not in d.name.lower():
        continue
      candidatos.append(d)
    else:
      d.close()
  if not candidatos:
    raise RuntimeError(
      "No encontre ningun mando.\n"
      "  - Conectalo por USB-C, o emparejalo por Bluetooth.\n"
      "  - Comprueba con: python probar_mando.py"
    )
  elegido = candidatos[0]
  for otro in candidatos[1:]:
    otro.close()
  return elegido


class MandoPS5:
  """Lee el mando en un hilo aparte y guarda el ultimo valor de cada eje."""

  def __init__(self, nombre_parcial: str | None = None):
    self.dev = buscar_mando(nombre_parcial)
    self.nombre = self.dev.name
    self.ruta = self.dev.path

    # Rango real de cada eje (el DualSense da 0..255, no -32768..32767).
    self._rango: dict[int, tuple[float, float]] = {}
    for code, info in self.dev.capabilities().get(ecodes.EV_ABS, []):
      self._rango[code] = (float(info.min), float(info.max))

    self._ejes: dict[int, float] = {}
    self._botones: set[int] = set()
    self._lock = threading.Lock()
    self._parar = threading.Event()

    self._hilo = threading.Thread(target=self._bucle, daemon=True)
    self._hilo.start()

  def _bucle(self) -> None:
    try:
      for ev in self.dev.read_loop():
        if self._parar.is_set():
          return
        if ev.type == ecodes.EV_ABS:
          with self._lock:
            self._ejes[ev.code] = float(ev.value)
        elif ev.type == ecodes.EV_KEY:
          with self._lock:
            if ev.value:
              self._botones.add(ev.code)
            else:
              self._botones.discard(ev.code)
    except OSError:
      # El mando se desconecto.
      return

  def cerrar(self) -> None:
    self._parar.set()
    try:
      self.dev.close()
    except Exception:
      pass

  # ---------------------------------------------------------------- lectura

  def eje(self, code: int) -> float:
    """Eje normalizado a -1..1, con zona muerta y centro en reposo."""
    lo, hi = self._rango.get(code, (0.0, 255.0))
    with self._lock:
      crudo = self._ejes.get(code)
    if crudo is None:
      return 0.0
    centro = (lo + hi) / 2.0
    amplitud = (hi - lo) / 2.0
    v = (crudo - centro) / amplitud if amplitud else 0.0
    v = max(-1.0, min(1.0, v))
    if abs(v) < ZONA_MUERTA:
      return 0.0
    # Reescala fuera de la zona muerta para no dar un salto al salir de ella.
    signo = 1.0 if v > 0 else -1.0
    return signo * (abs(v) - ZONA_MUERTA) / (1.0 - ZONA_MUERTA)

  def boton(self, code: int) -> bool:
    with self._lock:
      return code in self._botones

  @property
  def avance(self) -> float:
    """Stick izquierdo vertical. Arriba = +1 = adelante.

    MEDIDO en el DualSense de esta maquina (2026-09-29): ABS_Y CRECE hacia
    arriba, al contrario de la convencion habitual de evdev (donde arriba es
    el minimo). Por eso NO se niega. Si en otro mando sale al reves, se
    corrige al vuelo con --invertir-avance en jugar_mando.py.
    """
    return _sin_cero_negativo(self.eje(ecodes.ABS_Y))

  @property
  def lateral(self) -> float:
    """Stick izquierdo horizontal. Izquierda = +1 (+y del robot es su izquierda)."""
    return _sin_cero_negativo(-self.eje(ecodes.ABS_X))

  @property
  def giro(self) -> float:
    """Stick derecho horizontal. Izquierda = +1 (+yaw es antihorario desde arriba)."""
    return _sin_cero_negativo(-self.eje(ecodes.ABS_RX))

  @property
  def alto(self) -> bool:
    """Circulo: parada inmediata (comando cero)."""
    return self.boton(ecodes.BTN_EAST)


def escalar(v: float, lo: float, hi: float) -> float:
  """Lleva v (-1..1) al rango [lo, hi], que puede ser asimetrico."""
  return v * hi if v >= 0.0 else abs(v) * abs(lo) * -1.0
