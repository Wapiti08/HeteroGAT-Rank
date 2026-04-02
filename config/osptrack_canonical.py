"""
Canonical (QUT-aligned) extraction for OSPTrack rows.

This keeps the existing star-graph as a baseline elsewhere, but provides a
stronger ontology for multi-hop reasoning:

PKG -> PROC (LOAD)
PROC -> FILE (READ/WRITE/DELETE)
PROC -> NET  (CONNECT, DNS_QUERY/RESOLVE)
PROC -> CMD  (EXEC)

All events include a `phase` attribute: import vs install.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from utils.osptrack_decode import as_list, get_dict, iter_dicts, normalize_command
from utils.prostring import process_string


NodeType = Literal["PKG", "PROC", "FILE", "NET", "CMD", "SYSCALL"]
EdgeType = Literal[
    "LOAD",
    "READ",
    "WRITE",
    "DELETE",
    "CONNECT",
    "DNS_QUERY",
    "RESOLVE",
    "EXEC",
    "INVOKE",
    "DEPEND",
]
Phase = Literal["import", "install"]


@dataclass(frozen=True)
class NodeRef:
    type: NodeType
    key: str
    attrs: Dict[str, Any]


@dataclass(frozen=True)
class EdgeEvent:
    src: NodeRef
    etype: EdgeType
    dst: NodeRef
    attrs: Dict[str, Any]


def pkg_key(row: Dict[str, Any]) -> str:
    return f"{row['Ecosystem']}::{row['Name']}@{row['Version']}"


def proc_key(row: Dict[str, Any]) -> str:
    # Placeholder "runtime proc" to align with QUT's PKG->PROC hub.
    return f"{pkg_key(row)}::proc::install"


def _pkg_node(row: Dict[str, Any]) -> NodeRef:
    return NodeRef(
        type="PKG",
        key=pkg_key(row),
        attrs={"eco": row.get("Ecosystem"), "name": row.get("Name"), "version": row.get("Version")},
    )


def _proc_node(row: Dict[str, Any]) -> NodeRef:
    return NodeRef(type="PROC", key=proc_key(row), attrs={"role": "install"})


def extract_events(row: Dict[str, Any]) -> List[EdgeEvent]:
    """Extract canonical events from a single OSPTrack row (from label_data.pkl)."""
    events: List[EdgeEvent] = []
    pkg = _pkg_node(row)
    proc = _proc_node(row)

    # Always connect PKG->PROC so all evidence joins at PROC.
    events.append(EdgeEvent(src=pkg, etype="LOAD", dst=proc, attrs={"phase": "install"}))

    def add_file_events(col: str, phase: Phase) -> None:
        for ent in iter_dicts(row.get(col)):
            raw_path = get_dict(ent, "Path")
            if not raw_path:
                continue
            p = process_string(str(raw_path))
            fnode = NodeRef(type="FILE", key=f"{pkg.key}::file::{p}", attrs={"path": p})

            if ent.get("Read") is True:
                events.append(EdgeEvent(src=proc, etype="READ", dst=fnode, attrs={"phase": phase}))
            if ent.get("Write") is True:
                events.append(EdgeEvent(src=proc, etype="WRITE", dst=fnode, attrs={"phase": phase}))
            if ent.get("Delete") is True:
                events.append(EdgeEvent(src=proc, etype="DELETE", dst=fnode, attrs={"phase": phase}))

    def add_socket_events(col: str, phase: Phase) -> None:
        for ent in iter_dicts(row.get(col)):
            ip = get_dict(ent, "Address")
            port = get_dict(ent, "Port")
            hostnames = [h for h in as_list(get_dict(ent, "Hostnames", [])) if h]

            # Skip localhost-ish / empty
            if ip in (None, "", "::1"):
                ip = None
            if port == 0:
                port = None

            # Represent each endpoint as NET node; keep simple & alignable.
            if ip:
                n = NodeRef(type="NET", key=f"ip::{ip}", attrs={"kind": "ip", "ip": ip})
                events.append(EdgeEvent(src=proc, etype="CONNECT", dst=n, attrs={"phase": phase, "port": port}))
            for h in hostnames:
                n = NodeRef(type="NET", key=f"host::{h}", attrs={"kind": "host", "host": h})
                events.append(EdgeEvent(src=proc, etype="CONNECT", dst=n, attrs={"phase": phase, "port": port}))

    def add_dns_events(col: str, phase: Phase) -> None:
        for ent in iter_dicts(row.get(col)):
            queries = get_dict(ent, "Queries", [])
            for q in as_list(queries):
                if not isinstance(q, dict):
                    continue
                host = q.get("Hostname")
                if not host:
                    continue
                types = [str(t) for t in as_list(q.get("Types", [])) if t]
                n = NodeRef(type="NET", key=f"dns::{host}", attrs={"kind": "domain", "domain": host})
                events.append(EdgeEvent(src=proc, etype="DNS_QUERY", dst=n, attrs={"phase": phase, "dns_types": types}))
                # RESOLVE is optional; keep as a separate edge type to support ablation.
                events.append(EdgeEvent(src=proc, etype="RESOLVE", dst=n, attrs={"phase": phase}))

    def add_cmd_events(col: str, phase: Phase) -> None:
        for ent in iter_dicts(row.get(col)):
            cmd = normalize_command(get_dict(ent, "Command", []))
            if not cmd:
                continue
            c = process_string(cmd, max_len=500)
            n = NodeRef(type="CMD", key=f"cmd::{c}", attrs={"cmd": c})
            events.append(EdgeEvent(src=proc, etype="EXEC", dst=n, attrs={"phase": phase}))

    add_file_events("import_Files", "import")
    add_file_events("install_Files", "install")
    add_socket_events("import_Sockets", "import")
    add_socket_events("install_Sockets", "install")
    add_dns_events("import_DNS", "import")
    add_dns_events("install_DNS", "install")
    add_cmd_events("import_Commands", "import")
    add_cmd_events("install_Commands", "install")

    return events

