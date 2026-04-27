## Report-Backed OSP/PyPI Hunting Mapping Candidates

This table collects OSPTrack/PyPI packages that can be used for Reliability
Analysis when matching surfaced hunting evidence to public security reports.

| Package | Report Source | Reported Behavior | Surfaced Hunting Evidence | Mapping | Notes |
|---|---|---|---|---|---|
| `capmonstercloudclent`, `capmonstercloudclinent`, `capmonstercloudclieent`, `capmonstercloudclinet`, `capmonstercloudclouidclient`, `capmonsstercloudclient` | Phylum / Veracode / RH-ISAC report on the March 2024 PyPI typosquatting campaign | Typosquatted variants of `capmonstercloudclient`; malicious `setup.py`/install hooks delivered a payload associated with zgRAT/data stealing. | Generate OSP graphs from these package rows, then inspect install-phase `PROC|EXEC|CMD`, `PROC|CONNECT|NET`, and file-access evidence. | Direct if the generated graph surfaces install-time execution/payload endpoints; otherwise Partial. | Use the typosquat names, not the legitimate package name `capmonstercloudclient`. Veracode lists specific variants that overlap this repo's malicious package list. |
| `ligitgays` | Unit 42, "Six Malicious Python Packages in the PyPI Targeting Windows Users" | Malicious PyPI package using remote payload retrieval and W4SP-like credential/crypto-wallet theft behavior. | Candidate OSP graph should be checked for install-time network destinations, temporary file writes, and command execution. | Candidate / Direct if exact URL or payload execution evidence is surfaced. | The public report includes package name, malicious URL, author pattern, and setup-time execution behavior. |
| `bettercolors` | PyPI malicious-package / color-themed typosquatting reports | Reported as a malicious PyPI package in color/colorama-themed supply-chain activity. | Candidate OSP graph should be checked for install-time command execution, obfuscation-related file activity, and outbound network behavior. | Candidate / Partial | Use only if the generated graph provides behavior-level evidence; public reporting is weaker than the capmonstercloud and ligitgays cases. |

Suggested command target set:

```text
capmonstercloudclent,capmonstercloudclinent,capmonstercloudclieent,capmonstercloudclinet,capmonstercloudclouidclient,capmonsstercloudclient,ligitgays,bettercolors
```
