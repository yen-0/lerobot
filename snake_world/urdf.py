from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class JointSpec:
    name: str
    parent: str
    child: str
    joint_type: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: float | None = None
    upper: float | None = None


def _vec(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if not text:
        return np.array(default, dtype=np.float64)
    return np.array([float(x) for x in text.split()], dtype=np.float64)


def parse_urdf_joints(path: str | Path) -> dict[str, JointSpec]:
    root = ET.parse(path).getroot()
    joints: dict[str, JointSpec] = {}
    for node in root.findall("joint"):
        name = node.attrib["name"]
        parent = node.find("parent")
        child = node.find("child")
        if parent is None or child is None:
            continue
        origin = node.find("origin")
        axis = node.find("axis")
        limit = node.find("limit")
        joints[name] = JointSpec(
            name=name,
            parent=parent.attrib["link"],
            child=child.attrib["link"],
            joint_type=node.attrib.get("type", "fixed"),
            xyz=_vec(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)),
            rpy=_vec(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0)),
            axis=_vec(axis.attrib.get("xyz") if axis is not None else None, (0.0, 0.0, 1.0)),
            lower=float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else None,
            upper=float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else None,
        )
    return joints


def chain_from_links(joints: dict[str, JointSpec], root_link: str, tip_link: str) -> list[JointSpec]:
    by_parent: dict[str, list[JointSpec]] = {}
    for joint in joints.values():
        by_parent.setdefault(joint.parent, []).append(joint)

    stack: list[tuple[str, list[JointSpec]]] = [(root_link, [])]
    seen: set[str] = set()
    while stack:
        link, chain = stack.pop()
        if link == tip_link:
            return chain
        if link in seen:
            continue
        seen.add(link)
        for joint in by_parent.get(link, []):
            stack.append((joint.child, [*chain, joint]))
    raise ValueError(f"No URDF chain from {root_link!r} to {tip_link!r}")
