## Report-Backed Hunting Mapping Candidates

This table is intended as a draft source for the Reliability Analysis mapping between surfaced hunting evidence and public security reports.

| Package | Report Source | Reported Behavior | Surfaced Hunting Evidence | Mapping | Notes |
|---|---|---|---|---|---|
| `10Cent10` / `10Cent11` | JFrog Security, "Malicious packages in PyPI use stealthy exfiltration methods"; The Record, "Malicious Python packages caught stealing Discord tokens, installing shells" | Reported as malicious PyPI packages with a connectback shell to hardcoded address `104.248.19.57`. | QUT graphs exist for `10Cent10-999.0.4.tar.gz` and `10Cent11-999.0.4.tar.gz`; their install-phase events include network/socket-related behavior and command/file activity surfaced by hunting. | Partial | Use Direct only if the raw trace or graph preserves the specific IP/endpoint. Current canonical QUT graph keys may encode aggregate counts for some TCP fields, so avoid claiming the exact report IP from the graph alone. |
| `importantpackage` / `important-package` | JFrog Security, "Malicious packages in PyPI use stealthy exfiltration methods" | Hidden connectback shell using TrevorC2 and C2-like HTTP communication through `psec.forward.io.global.prod.fastly.net`. | Candidate independent case if a graph can be generated from a matching package artifact or OSP/QUT record. | Candidate | Good example for mapping network C2 evidence, subprocess execution, and install/runtime payload behavior. |
| `pptest` / `ipboards` | JFrog Security, "Malicious packages in PyPI use stealthy exfiltration methods" | DNS tunneling and host/user/path exfiltration through attacker-controlled domains. | Candidate independent case if graph data contains DNS/domain nodes or network destinations. | Candidate | Useful when hunting evidence includes DNS/domain edges rather than only ports. |
| `owlmoon` / `DiscordSafety` / `yiffparty` | JFrog Security, "Malicious packages in PyPI use stealthy exfiltration methods" | Discord token stealer behavior and exfiltration to webhook or attacker endpoint. | Candidate independent case if graph data exposes suspicious HTTP destinations, sensitive file reads, or token-store access. | Candidate | Best for explaining partial mappings from file access and outbound network evidence to credential theft behavior. |
| `ligitgays` / `xboxredeemer` / `syntax-init` / `xboxlivepy` / `Ligitkidss` / `tls-python` | Unit 42, "Six Malicious Python Packages in the PyPI Targeting Windows Users" | Installation/execution triggers remote payload retrieval, temporary file creation/execution, and W4SP-like credential/crypto wallet theft. | Candidate independent case if matching graphs can be generated; `ligitgays` is present in `cred/mal_packages_with_desc.csv`. | Candidate | Strong public report with concrete package names, malicious URLs, and setup-time execution behavior. |

Suggested mapping labels:

- Direct: surfaced evidence matches a report artifact at the same granularity, such as the same IP, domain, URL, webhook, or command.
- Partial: surfaced evidence supports the same behavior class, such as install-time network activity or shell execution, but lacks the exact IOC.
- Candidate: public report is strong, but matching local graph evidence still needs to be generated or verified.
