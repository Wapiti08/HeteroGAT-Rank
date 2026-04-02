"""
Canonical (QUT-aligned) extraction for QUT-DV25 processed rows.

Each QUT processed row is package-centric. We emit a shared set of canonical
events (`EdgeEvent`) so the same graph builder and model backbone can be reused
across datasets.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from config.osptrack_canonical import EdgeEvent, NodeRef
from utils.qut_decode import parse_listish


QUTKind = Literal["syscall", "opensnoop", "filetop", "install", "tcp", "pattern"]


def qut_pkg_node(package_name: str) -> NodeRef:
    # QUT packages are python tarballs; version parsing is messy, so keep it in attrs.
    return NodeRef(type="PKG", key=f"qut::{package_name}", attrs={"eco": "pypi", "package": package_name})


def qut_proc_node(package_name: str) -> NodeRef:
    return NodeRef(type="PROC", key=f"qut::{package_name}::proc::install", attrs={"role": "install"})


def extract_events_from_row(row: Dict[str, Any], kind: QUTKind) -> List[EdgeEvent]:
    """Extract canonical events for one QUT processed CSV row."""
    pkg_name = str(row.get("Package_Name", "")).strip()
    if not pkg_name:
        return []

    pkg = qut_pkg_node(pkg_name)
    proc = qut_proc_node(pkg_name)
    events: List[EdgeEvent] = []

    # Always include LOAD backbone so multi-table joins connect.
    events.append(EdgeEvent(src=pkg, etype="LOAD", dst=proc, attrs={"phase": "install"}))

    if kind == "install":
        # Dependency edges: PKG -> PKG (DEPEND)
        direct = parse_listish(row.get("Direct_Dependencies"))
        indirect = parse_listish(row.get("Indirect_Dependencies"))
        total = parse_listish(row.get("Total_Dependencies"))
        deps = direct or total or []
        # Prefer direct deps, but keep a few indirect as well if present.
        for dep in deps:
            dep_pkg = NodeRef(type="PKG", key=f"qut::{dep}", attrs={"eco": "pypi", "package": dep})
            events.append(EdgeEvent(src=pkg, etype="DEPEND", dst=dep_pkg, attrs={"scope": "direct"}))
        for dep in indirect[:50]:
            dep_pkg = NodeRef(type="PKG", key=f"qut::{dep}", attrs={"eco": "pypi", "package": dep})
            events.append(EdgeEvent(src=pkg, etype="DEPEND", dst=dep_pkg, attrs={"scope": "indirect"}))
        return events

    if kind == "syscall":
        # Unique_System_Calls_List is a comma-separated string in this dataset.
        calls = parse_listish(row.get("Unique_System_Calls_List"))
        for c in calls:
            sc = NodeRef(type="SYSCALL", key=f"syscall::{c}", attrs={"name": c})
            events.append(EdgeEvent(src=proc, etype="INVOKE", dst=sc, attrs={"phase": "install"}))
        return events

    if kind == "opensnoop":
        # Directory bucket features: treat as FILE nodes.
        buckets = [
            ("Root_DIR_Installation", "ROOT_DIR"),
            ("Temporary_DIR_Installation", "TMP_DIR"),
            ("Home_DIR_Installation", "HOME_DIR"),
            ("Etc_DIR_Installation", "ETC_DIR"),
            ("Other_DIR_Installation", "OTHER_DIR"),
        ]
        for col, b in buckets:
            v = row.get(col)
            try:
                n = float(v)
            except Exception:
                n = 0.0
            if n <= 0:
                continue
            f = NodeRef(type="FILE", key=f"{pkg.key}::bucket::{b}", attrs={"bucket": b})
            # Opensnoop is largely file opens; map to WRITE as a generic access (can ablate).
            events.append(EdgeEvent(src=proc, etype="WRITE", dst=f, attrs={"phase": "install", "count": n}))
        return events

    if kind == "filetop":
        # Use aggregate read/write transfer as edges to an ALL_FILES bucket.
        all_files = NodeRef(type="FILE", key=f"{pkg.key}::bucket::ALL_FILES", attrs={"bucket": "ALL_FILES"})
        rbytes = row.get("Total_Read_Data_Transfer")
        wbytes = row.get("Total_Write_Data_Transfer")
        try:
            rb = float(rbytes)
        except Exception:
            rb = 0.0
        try:
            wb = float(wbytes)
        except Exception:
            wb = 0.0
        if rb > 0:
            events.append(EdgeEvent(src=proc, etype="READ", dst=all_files, attrs={"phase": "install", "bytes": rb}))
        if wb > 0:
            events.append(EdgeEvent(src=proc, etype="WRITE", dst=all_files, attrs={"phase": "install", "bytes": wb}))

        # Process lists (stringified) become PROC nodes linked via EXEC.
        for col in ["Read_Processes", "Write_Processes", "File_Access_Processes"]:
            for p in parse_listish(row.get(col)):
                pn = NodeRef(type="PROC", key=f"{pkg.key}::proc::{p}", attrs={"name": p})
                events.append(EdgeEvent(src=proc, etype="EXEC", dst=pn, attrs={"phase": "install", "source": col}))
        return events

    if kind == "tcp":
        # IP/port access lists become NET nodes with CONNECT edges.
        for ip in parse_listish(row.get("Remote_IP_Address_Access")):
            n = NodeRef(type="NET", key=f"ip::{ip}", attrs={"kind": "ip", "ip": ip})
            events.append(EdgeEvent(src=proc, etype="CONNECT", dst=n, attrs={"phase": "install"}))
        for port in parse_listish(row.get("Remote_Port_Access")):
            n = NodeRef(type="NET", key=f"port::{port}", attrs={"kind": "port", "port": port})
            events.append(EdgeEvent(src=proc, etype="CONNECT", dst=n, attrs={"phase": "install"}))
        return events

    if kind == "pattern":
        # Pattern_i are already engineered strings. Represent as CMD nodes executed.
        for i in range(1, 11):
            col = f"Pattern_{i}"
            v = row.get(col)
            if v is None:
                continue
            s = str(v).strip()
            if not s or s.lower() == "nan":
                continue
            cmd = NodeRef(type="CMD", key=f"pattern::{i}::{s}", attrs={"pattern_id": i, "pattern": s})
            events.append(EdgeEvent(src=proc, etype="EXEC", dst=cmd, attrs={"phase": "install"}))
        return events

    return events

