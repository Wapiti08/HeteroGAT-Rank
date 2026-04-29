## Case studies (auto-generated)

Config: k=20 rarity_lambda=0.5 idf_cap=3.0 rarity_etypes=['PROC|CONNECT|NET', 'PROC|EXEC|CMD']

### Case 1: rubygems::commonmarker_pluggable@0.3.0.graph

- **graph**: `artifacts/osp_all/rubygems::commonmarker_pluggable@0.3.0.graph.pt`
- **label (y/Level)**: `1`
- **triage rank (lower is more suspicious)**: base `1243` -> rarity `232`

#### Top-K suspicious edges (base)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::commonmarker_pluggable@0.3.0::proc::install` | `FILE:rubygems::commonmarker_pluggable@0.3.0::file::default/fcntl-1.0.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::commonmarker_pluggable@0.3.0::proc::install` | `FILE:rubygems::commonmarker_pluggable@0.3.0::file::default/net-ftp-0.1.2.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::commonmarker_pluggable@0.3.0::proc::install` | `FILE:rubygems::commonmarker_pluggable@0.3.0::file::default/bundler-2.2.22.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::commonmarker_pluggable@0.3.0::proc::install` | `FILE:rubygems::commonmarker_pluggable@0.3.0::file::default/cgi-0.2.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::commonmarker_pluggable@0.3.0::proc::install` | `FILE:rubygems::commonmarker_pluggable@0.3.0::file::default/csv-3.1.9.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::commonmarker_pluggable@0.3.0::proc::install` | `FILE:rubygems::commonmarker_pluggable@0.3.0::file::default/date-3.1.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::commonmarker_pluggable@0.3.0::proc::install` | `FILE:rubygems::commonmarker_pluggable@0.3.0::file::default/dbm-1.1.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::commonmarker_pluggable@0.3.0::proc::install` | `FILE:rubygems::commonmarker_pluggable@0.3.0::file::default/debug-0.1.0.gemspec` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:rubygems::commonmarker_pluggable@0.3.0::proc::install` | `FILE:rubygems::commonmarker_pluggable@0.3.0::file::socket:[7]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:rubygems::commonmarker_pluggable@0.3.0::proc::install` | `FILE:rubygems::commonmarker_pluggable@0.3.0::file::anon_inode:[eventfd]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:rubygems::commonmarker_pluggable@0.3.0::proc::install` | `FILE:rubygems::commonmarker_pluggable@0.3.0::file::socket:[11]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:rubygems::commonmarker_pluggable@0.3.0::proc::install` | `FILE:rubygems::commonmarker_pluggable@0.3.0::file::host:[5]` |

#### Top-K suspicious edges (rarity-adjusted)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| 0.5002 | `PROC|CONNECT|NET` | `PROC:rubygems::commonmarker_pluggable@0.3.0::proc::install` | `NET:ip::8.8.4.4` |
| 0.5000 | `PROC|EXEC|CMD` | `PROC:rubygems::commonmarker_pluggable@0.3.0::proc::install` | `CMD:cmd::rubybin/analyze-ruby.rb--version0.3.0installcommonmarker_pluggable` |
| 0.5000 | `PROC|EXEC|CMD` | `PROC:rubygems::commonmarker_pluggable@0.3.0::proc::install` | `CMD:cmd::rubybin/geminstall-v0.3.0commonmarker_pluggable` |
| 0.5000 | `PROC|EXEC|CMD` | `PROC:rubygems::commonmarker_pluggable@0.3.0::proc::install` | `CMD:cmd::geminstall-v0.3.0commonmarker_pluggable` |
| 0.0750 | `PROC|CONNECT|NET` | `PROC:rubygems::commonmarker_pluggable@0.3.0::proc::install` | `NET:ip::151.101.65.227` |
| 0.0750 | `PROC|CONNECT|NET` | `PROC:rubygems::commonmarker_pluggable@0.3.0::proc::install` | `NET:ip::151.101.193.227` |
| 0.0750 | `PROC|CONNECT|NET` | `PROC:rubygems::commonmarker_pluggable@0.3.0::proc::install` | `NET:ip::151.101.129.227` |
| 0.0750 | `PROC|CONNECT|NET` | `PROC:rubygems::commonmarker_pluggable@0.3.0::proc::install` | `NET:ip::151.101.1.227` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::commonmarker_pluggable@0.3.0::proc::install` | `FILE:rubygems::commonmarker_pluggable@0.3.0::file::default/fcntl-1.0.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::commonmarker_pluggable@0.3.0::proc::install` | `FILE:rubygems::commonmarker_pluggable@0.3.0::file::default/net-ftp-0.1.2.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::commonmarker_pluggable@0.3.0::proc::install` | `FILE:rubygems::commonmarker_pluggable@0.3.0::file::default/bundler-2.2.22.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::commonmarker_pluggable@0.3.0::proc::install` | `FILE:rubygems::commonmarker_pluggable@0.3.0::file::default/cgi-0.2.0.gemspec` |

### Case 2: rubygems::ach-client@1.0.3.graph

- **graph**: `artifacts/osp_all/rubygems::ach-client@1.0.3.graph.pt`
- **label (y/Level)**: `1`
- **triage rank (lower is more suspicious)**: base `1244` -> rarity `375`

#### Top-K suspicious edges (base)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::ach-client@1.0.3::proc::install` | `FILE:rubygems::ach-client@1.0.3::file::default/fcntl-1.0.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::ach-client@1.0.3::proc::install` | `FILE:rubygems::ach-client@1.0.3::file::default/bigdecimal-3.0.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::ach-client@1.0.3::proc::install` | `FILE:rubygems::ach-client@1.0.3::file::default/etc-1.2.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::ach-client@1.0.3::proc::install` | `FILE:rubygems::ach-client@1.0.3::file::default/erb-2.2.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::ach-client@1.0.3::proc::install` | `FILE:rubygems::ach-client@1.0.3::file::default/english-0.7.1.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::ach-client@1.0.3::proc::install` | `FILE:rubygems::ach-client@1.0.3::file::default/drb-2.0.4.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::ach-client@1.0.3::proc::install` | `FILE:rubygems::ach-client@1.0.3::file::default/digest-3.0.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::ach-client@1.0.3::proc::install` | `FILE:rubygems::ach-client@1.0.3::file::default/did_you_mean-1.5.0.gemspec` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:rubygems::ach-client@1.0.3::proc::install` | `FILE:rubygems::ach-client@1.0.3::file::socket:[10]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:rubygems::ach-client@1.0.3::proc::install` | `FILE:rubygems::ach-client@1.0.3::file::host:[5]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:rubygems::ach-client@1.0.3::proc::install` | `FILE:rubygems::ach-client@1.0.3::file::anon_inode:[eventfd]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:rubygems::ach-client@1.0.3::proc::install` | `FILE:rubygems::ach-client@1.0.3::file::socket:[6]` |

#### Top-K suspicious edges (rarity-adjusted)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| 0.5000 | `PROC|EXEC|CMD` | `PROC:rubygems::ach-client@1.0.3::proc::install` | `CMD:cmd::rubybin/analyze-ruby.rb--version1.0.3installach-client` |
| 0.5000 | `PROC|EXEC|CMD` | `PROC:rubygems::ach-client@1.0.3::proc::install` | `CMD:cmd::rubybin/geminstall-v1.0.3ach-client` |
| 0.5000 | `PROC|EXEC|CMD` | `PROC:rubygems::ach-client@1.0.3::proc::install` | `CMD:cmd::geminstall-v1.0.3ach-client` |
| 0.0750 | `PROC|CONNECT|NET` | `PROC:rubygems::ach-client@1.0.3::proc::install` | `NET:ip::151.101.1.227` |
| 0.0750 | `PROC|CONNECT|NET` | `PROC:rubygems::ach-client@1.0.3::proc::install` | `NET:ip::151.101.129.227` |
| 0.0750 | `PROC|CONNECT|NET` | `PROC:rubygems::ach-client@1.0.3::proc::install` | `NET:ip::151.101.193.227` |
| 0.0750 | `PROC|CONNECT|NET` | `PROC:rubygems::ach-client@1.0.3::proc::install` | `NET:ip::151.101.65.227` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::ach-client@1.0.3::proc::install` | `FILE:rubygems::ach-client@1.0.3::file::default/fcntl-1.0.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::ach-client@1.0.3::proc::install` | `FILE:rubygems::ach-client@1.0.3::file::default/bigdecimal-3.0.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::ach-client@1.0.3::proc::install` | `FILE:rubygems::ach-client@1.0.3::file::default/etc-1.2.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::ach-client@1.0.3::proc::install` | `FILE:rubygems::ach-client@1.0.3::file::default/erb-2.2.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::ach-client@1.0.3::proc::install` | `FILE:rubygems::ach-client@1.0.3::file::default/english-0.7.1.gemspec` |

### Case 3: rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0.graph

- **graph**: `artifacts/osp_all/rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0.graph.pt`
- **label (y/Level)**: `1`
- **triage rank (lower is more suspicious)**: base `1245` -> rarity `376`

#### Top-K suspicious edges (base)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::proc::install` | `FILE:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::file::default/fcntl-1.0.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::proc::install` | `FILE:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::file::default/bigdecimal-3.0.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::proc::install` | `FILE:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::file::default/etc-1.2.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::proc::install` | `FILE:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::file::default/erb-2.2.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::proc::install` | `FILE:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::file::default/english-0.7.1.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::proc::install` | `FILE:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::file::default/drb-2.0.4.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::proc::install` | `FILE:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::file::default/digest-3.0.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::proc::install` | `FILE:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::file::default/did_you_mean-1.5.0.gemspec` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::proc::install` | `FILE:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::file::socket:[10]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::proc::install` | `FILE:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::file::host:[5]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::proc::install` | `FILE:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::file::anon_inode:[eventfd]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::proc::install` | `FILE:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::file::socket:[6]` |

#### Top-K suspicious edges (rarity-adjusted)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| 0.5000 | `PROC|EXEC|CMD` | `PROC:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::proc::install` | `CMD:cmd::rubybin/analyze-ruby.rb--version0.1.0installfluent_plugin-cloudwatch-logs-foxtrot9` |
| 0.5000 | `PROC|EXEC|CMD` | `PROC:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::proc::install` | `CMD:cmd::rubybin/geminstall-v0.1.0fluent_plugin-cloudwatch-logs-foxtrot9` |
| 0.5000 | `PROC|EXEC|CMD` | `PROC:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::proc::install` | `CMD:cmd::geminstall-v0.1.0fluent_plugin-cloudwatch-logs-foxtrot9` |
| 0.0750 | `PROC|CONNECT|NET` | `PROC:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::proc::install` | `NET:ip::151.101.1.227` |
| 0.0750 | `PROC|CONNECT|NET` | `PROC:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::proc::install` | `NET:ip::151.101.129.227` |
| 0.0750 | `PROC|CONNECT|NET` | `PROC:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::proc::install` | `NET:ip::151.101.193.227` |
| 0.0750 | `PROC|CONNECT|NET` | `PROC:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::proc::install` | `NET:ip::151.101.65.227` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::proc::install` | `FILE:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::file::default/fcntl-1.0.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::proc::install` | `FILE:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::file::default/bigdecimal-3.0.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::proc::install` | `FILE:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::file::default/etc-1.2.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::proc::install` | `FILE:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::file::default/erb-2.2.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::proc::install` | `FILE:rubygems::fluent_plugin-cloudwatch-logs-foxtrot9@0.1.0::file::default/english-0.7.1.gemspec` |
