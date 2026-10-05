# Pentabot – modelo MuJoCo (MJCF)

Generado automáticamente desde `CuerpoBase.iam` (Inventor 2027). MuJoCo ≥ 3.1 (probado en 3.14).

```bash
pip install mujoco
python -m mujoco.viewer --mjcf scene.xml        # Ctrl+tecla de keyframe: "home" (de pie) o "cad" (pose del CAD)
```

`pentabot.xml` contiene el robot y `scene.xml` agrega el suelo y la luz (usa este para entrenar). `assets/` tiene las mallas STL en mm (la escala 0.001 está en el XML).

## Estructura
- `torso` (freejoint) + 5 piernas `L1…L5`, en sentido antihorario visto desde arriba. L1 está a ~37° del eje +x.
- Cada pierna tiene 3 articulaciones y 3 actuadores `position`, con los mismos nombres:
  - `Lk_hip_yaw` (coxa, eje vertical; + = antihorario)
  - `Lk_hip_pitch` (fémur; + = sube el fémur)
  - `Lk_knee` (tibia; + = sube la tibia)
- **q = 0** es la pose de apoyo del CAD (fémur horizontal, la de la pierna L1 = `Hombro:2`). La altura del torso de pie es 0.199 m y los pies quedan a un radio de 0.344 m.
- Correspondencia con Inventor: L1=Hombro:2, L2=Hombro:1, L3=Hombro:5, L4=Hombro:4, L5=Hombro:3. Para extensionUnida:k y pataLarga:k el índice es el mismo.
- Longitudes:
  - Coxa: 45.3 mm (eje J1→J2, horizontal)
  - Fémur: 120.0 mm
  - Tibia: 181.1 mm (J3→centro del pie)
  - Pie: esfera R = 18 mm
- Sensores:
  - IMU en el torso (`imu_quat`, `imu_gyro`, `imu_acc`, `imu_vel`)
  - `*_pos`, `*_vel` y `*_tau` por articulación
  - `Lk_foot_touch` por pie

## Cómo se calculó (y supuestos)
| Elemento | Modelo | Fuente |
|---|---|---|
| Geometría / ejes | transformaciones exactas de las 203 piezas del ensamble; eje = horn del servo, centro = punto medio horn–rodamiento 6901 | CAD |
| PLA impreso | cáscara 0.8 mm (ρ = 1240) + relleno 17.5 % integrado sobre la malla (sólido + superficie) | supuesto: 2 perímetros, relleno 15–20 % |
| Pie | núcleo PLA + casquete TPU (ρ = 1210), mismo modelo de cáscara | CAD |
| Acrílico | sólido, ρ = 1190 | – |
| MG995 | 55 g, 0.981 N·m de bloqueo y 6.54 rad/s en vacío @6 V | datasheet TowerPro |
| 6901, tornillos M3x45 y tuercas | 11 g; acero ρ = 7900 | SKF / CAD |
| **Electrónica + batería** | **0.18 kg (PLACEHOLDER)**, caja de 10×6×3 cm al centro del cuerpo | **sin dato, medir** |

Masas por link: torso 0.915 kg (incluye el placeholder), coxa 0.115, fémur 0.051 y tibia 0.144. **Total: 2.465 kg.**

**Servo:**
- `kp = 11.24 N·m/rad`: satura a ~5° de error.
- `forcerange = ±0.981 N·m`.
- `damping = 0.150 N·m·s`: τ_stall/ω₀, reproduce la recta par-velocidad del motor.
- `armature = 0.005` y `frictionloss = 0.03`: estimados.
- `range = ±90°` alrededor de q=0.

**Contacto:**
- Pies: esfera con `condim=6`, μ = 0.9 y `solref 0.005 1`.
- Resto del robot: cascos convexos que solo chocan con el suelo. No hay auto-colisión.
- Con el peso del robot, el TPU 95A se deforma ~0.1 mm (Hertz), así que en la práctica el contacto es rígido. Lo que aporta el TPU es fricción, no amortiguación.

**Simulación:** `timestep 0.002`, `implicitfast`, cono elíptico, `impratio 10`. Corre a ~7 600 pasos/s (15× tiempo real) en 1 CPU.

## Validación (`validate.py`)
1. **Cinemática directa en pose CAD contra Inventor (centro de cada pie):**
   - Error de 0.0005 mm en las piernas en apoyo.
   - Error de 0.2–0.5 mm en las dos piernas levantadas, porque las restricciones de Inventor no son exactas.
   - Los ejes son coherentes entre las 5 piernas (residuo de rotación < 1e-12).
2. **De pie, 3 s:** el robot es estable, con inclinación de 0.008°.
   - Reparto de carga de 4.84 N por pie, que suma exactamente el peso.
   - Par en hip_pitch: 0.21 N·m (21 % del bloqueo).
3. **Apoyo en 3 patas:** el par de hip_pitch sube a 0.63 N·m (**64 % del bloqueo**) y el de knee a 0.50 N·m.
4. **20 s con acciones aleatorias:** sin NaN ni avisos del solver.

## Antes de confiar en el sim-to-real (en orden de impacto)
1. **Pesar el robot completo y la batería.** Corrige `M_ELEC` en `build.py` y regenera. Este es el error más grande que queda.
2. **Margen de par:** con 3 patas de apoyo, el MG995 trabaja al 64 % del bloqueo, y en una marcha dinámica se satura. En el entrenamiento penaliza el par, o no confíes en marchas con trípode rápido.
3. **Identificar el servo (sysid):** cuelga una pierna, da escalones de posición y ajusta `kp`, `damping`, `armature` y `frictionloss` hasta que la respuesta del sim se parezca a la real. Los MG995 tienen juego en los engranajes de 1–2°, que no está modelado; agrégalo como ruido o retardo en la observación.
4. **Ceros y sentidos de los servos:** q=0 no es el pulso de 1500 µs. Mide para cada servo el ángulo real que da 1500 µs y su sentido, y haz la conversión en el controlador.
5. **Límites articulares reales** (choques mecánicos): ajusta `range`.
6. Usa aleatorización de dominio en masa (±15 %), fricción (0.6–1.1), `kp` y la latencia del bus PCA9685.

## Regenerar
1. Ejecuta `inventor_dump.py` en Inventor con CuerpoBase.iam activo. Genera `leaves.json` y `meshes/`.
2. Corre `python build.py && python assemble.py && python keyframes.py && python validate.py`.

Archivos faltantes en el ensamble que **no** están en el modelo: `ENSAMBLE_PATABOLA.ipt` y `PLACA_PDM.iam`, que apuntan a la unidad G:.
