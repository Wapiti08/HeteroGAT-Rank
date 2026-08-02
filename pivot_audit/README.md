# pivot_audit

Core module for constructing and validating runtime-pivot ground truth from:

1. trusted malicious-package labels;
2. independent IOC/TTP reports; and
3. sandbox runtime telemetry.

The package is split by responsibility:

- `schema.py`: immutable domain model and annotation columns.
- `normalization.py`: pure identity, IOC, and version normalization.
- `sources.py`: adapters for malicious-packages-info and Backstabbers.
- `osptrack.py`: OSPTrack telemetry decoding only.
- `triggers.py`: report-backed trigger extraction, provenance, and phase matching.
- `behaviors.py`: report-backed command/process/file/network behavior extraction
  and conservative telemetry support checks.
- `data/trigger_reference_evidence.json`: curated, reviewable facts extracted from
  campaign reports whose trigger text is not present in the local advisory JSON.
- `matching.py`: evidence-to-telemetry matching policy.
- `candidates.py`: candidate IDs and review priority.
- `osptrack_builder.py`: OSPTrack workflow orchestration.
- `review_queue.py`: confidence policy and duplicate-free human-review tasks.
- `reporting.py`: single-file output writing.
- `validation.py`: human annotation validation and verified export.
- `osptrack_cli.py`: OSPTrack command-line composition root.
- `qut.py`: QUT-DV25 dataset adapter and workflow.
- `core.py`: small compatibility facade; it contains no business logic.

`qut.py` provides the QUT-DV25 adapter. It treats TCP IPs as exactly
observable, file/string evidence as only partially observable through aggregate
features, and domains/URLs as unobservable in the processed QUT representation.

The module never reclassifies OSPTrack packages. Its unit of ground truth is a
runtime edge/event/sequence used as an analyst pivot.

## Trigger evidence

Trigger auditing keeps four evidence concepts in the same review row:

1. `reported_trigger`: `install`, `import`, `runtime`, `function_call`, `test`,
   or `unknown`, extracted from an external source;
2. `trigger_evidence` and `trigger_source_reference`: the reviewable reason and
   citation for that extraction;
3. `observed_trigger` and `observed_trigger_evidence`: the OSPTrack/QUT phase
   actually exercised by the sandbox;
4. `trigger_match_auto`: an automatic comparison that does not overwrite
   `manual_trigger_exercised`.

Missing evidence is represented explicitly as `unknown`; generic compromise
boilerplate such as "installed or running" is not treated as a causal trigger.

## Generated review files

Each runtime dataset produces exactly one file:

- `ground_truth/osptrack_pivot_gt/review.tsv`
- `ground_truth/qut_pivot_gt/review.tsv`

One row represents one sandbox execution plus one external report. Multiple IOCs
and behaviors from the same report are stored together in `reported_iocs` and
`reported_behaviors`, avoiding repeated trigger and trace review. Concrete
artifacts and runtime behavior use separate assessments in the same row:

- artifact IOCs: IP, domain, URL, endpoint/port, payload filename, string, and
  file evidence;
- behavior events: payload download, process execution, sensitive-data access,
  archive collection, exfiltration, persistence, reverse shell, backdoor,
  denial of service, and propagation.

Package identity is never treated as an IOC. A behavior enters the queue only
when it has high-confidence report provenance, an exact version mapping, and a
matched sandbox trigger. Behavior support is conservative: commands such as
`curl`/`wget`, non-harness processes, sensitive file reads, archive writes, and
persistence paths can support a behavior, while semantic claims such as
exfiltration remain partial without data-flow confirmation.

Every item in the two JSON columns also carries `attribution_scope`. A
`record_attached` IOC was read directly from a package-specific malware record,
but the source may still describe a campaign-wide IOC set rather than prove that
every IOC applies to every listed package version. `package_row` is tied to a
Backstabbers package row; `referenced_report` and `package_row_or_reference`
identify behavior summarized from an external report. This provenance is kept
separate from whether the sandbox actually observed the item.

`confidence=high` requires an exact version mapping, a high-confidence reported
trigger, an observed trigger match, structured IOC extraction, and telemetry that
can observe every IOC type exactly. Any missing condition produces `low`, with
the reason recorded in `determination_basis`. Negative or partial automatic
results remain marked `manual_review_required=yes`.

Only five reviewer fields remain: `review_decision`, `review_matching_event`,
`review_status`, `reviewer`, and `review_note`. Regeneration preserves them by
stable `review_id`.
