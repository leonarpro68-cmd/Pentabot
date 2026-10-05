"""Visor interactivo: arranca en la keyframe 'home' (de pie) con los servos sosteniendo la pose."""
import os, mujoco, mujoco.viewer

XML = os.path.join(os.path.dirname(__file__), "..", "model", "scene.xml")
m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
# El panel derecho "Control" tiene un slider por servo (rad). Ctrl+L recarga si editas el XML.
mujoco.viewer.launch(m, d)
