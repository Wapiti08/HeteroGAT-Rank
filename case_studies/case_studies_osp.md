## Case studies (auto-generated)

Config: k=20 rarity_lambda=0.5 idf_cap=3.0 rarity_etypes=['PROC|CONNECT|NET', 'PROC|EXEC|CMD']

### Case 1: rubygems::log4r_logstash@0.1.1.graph

- **graph**: `artifacts/osp_all/rubygems::log4r_logstash@0.1.1.graph.pt`
- **label (y/Level)**: `1`
- **triage rank (lower is more suspicious)**: base `1207` -> rarity `362`

#### Top-K suspicious edges (base)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::log4r_logstash@0.1.1::proc::install` | `FILE:rubygems::log4r_logstash@0.1.1::file::default/fcntl-1.0.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::log4r_logstash@0.1.1::proc::install` | `FILE:rubygems::log4r_logstash@0.1.1::file::default/bigdecimal-3.0.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::log4r_logstash@0.1.1::proc::install` | `FILE:rubygems::log4r_logstash@0.1.1::file::default/etc-1.2.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::log4r_logstash@0.1.1::proc::install` | `FILE:rubygems::log4r_logstash@0.1.1::file::default/erb-2.2.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::log4r_logstash@0.1.1::proc::install` | `FILE:rubygems::log4r_logstash@0.1.1::file::default/english-0.7.1.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::log4r_logstash@0.1.1::proc::install` | `FILE:rubygems::log4r_logstash@0.1.1::file::default/drb-2.0.4.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::log4r_logstash@0.1.1::proc::install` | `FILE:rubygems::log4r_logstash@0.1.1::file::default/digest-3.0.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::log4r_logstash@0.1.1::proc::install` | `FILE:rubygems::log4r_logstash@0.1.1::file::default/did_you_mean-1.5.0.gemspec` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:rubygems::log4r_logstash@0.1.1::proc::install` | `FILE:rubygems::log4r_logstash@0.1.1::file::socket:[12]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:rubygems::log4r_logstash@0.1.1::proc::install` | `FILE:rubygems::log4r_logstash@0.1.1::file::host:[5]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:rubygems::log4r_logstash@0.1.1::proc::install` | `FILE:rubygems::log4r_logstash@0.1.1::file::anon_inode:[eventfd]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:rubygems::log4r_logstash@0.1.1::proc::install` | `FILE:rubygems::log4r_logstash@0.1.1::file::socket:[7]` |

#### Top-K suspicious edges (rarity-adjusted)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| 0.5002 | `PROC|CONNECT|NET` | `PROC:rubygems::log4r_logstash@0.1.1::proc::install` | `NET:host::index.rubygems.org` |
| 0.5002 | `PROC|CONNECT|NET` | `PROC:rubygems::log4r_logstash@0.1.1::proc::install` | `NET:host::rubygems.org` |
| 0.5000 | `PROC|EXEC|CMD` | `PROC:rubygems::log4r_logstash@0.1.1::proc::install` | `CMD:cmd::rubybin/analyze-ruby.rb--version0.1.1installlog4r_logstash` |
| 0.5000 | `PROC|EXEC|CMD` | `PROC:rubygems::log4r_logstash@0.1.1::proc::install` | `CMD:cmd::rubybin/geminstall-v0.1.1log4r_logstash` |
| 0.5000 | `PROC|EXEC|CMD` | `PROC:rubygems::log4r_logstash@0.1.1::proc::install` | `CMD:cmd::geminstall-v0.1.1log4r_logstash` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::log4r_logstash@0.1.1::proc::install` | `FILE:rubygems::log4r_logstash@0.1.1::file::default/fcntl-1.0.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::log4r_logstash@0.1.1::proc::install` | `FILE:rubygems::log4r_logstash@0.1.1::file::default/bigdecimal-3.0.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::log4r_logstash@0.1.1::proc::install` | `FILE:rubygems::log4r_logstash@0.1.1::file::default/etc-1.2.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::log4r_logstash@0.1.1::proc::install` | `FILE:rubygems::log4r_logstash@0.1.1::file::default/erb-2.2.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::log4r_logstash@0.1.1::proc::install` | `FILE:rubygems::log4r_logstash@0.1.1::file::default/english-0.7.1.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::log4r_logstash@0.1.1::proc::install` | `FILE:rubygems::log4r_logstash@0.1.1::file::default/drb-2.0.4.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::log4r_logstash@0.1.1::proc::install` | `FILE:rubygems::log4r_logstash@0.1.1::file::default/digest-3.0.0.gemspec` |

### Case 2: rubygems::alphabet_rocker@0.1.1.graph

- **graph**: `artifacts/osp_all/rubygems::alphabet_rocker@0.1.1.graph.pt`
- **label (y/Level)**: `1`
- **triage rank (lower is more suspicious)**: base `1208` -> rarity `363`

#### Top-K suspicious edges (base)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::alphabet_rocker@0.1.1::proc::install` | `FILE:rubygems::alphabet_rocker@0.1.1::file::default/fcntl-1.0.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::alphabet_rocker@0.1.1::proc::install` | `FILE:rubygems::alphabet_rocker@0.1.1::file::default/bigdecimal-3.0.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::alphabet_rocker@0.1.1::proc::install` | `FILE:rubygems::alphabet_rocker@0.1.1::file::default/etc-1.2.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::alphabet_rocker@0.1.1::proc::install` | `FILE:rubygems::alphabet_rocker@0.1.1::file::default/erb-2.2.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::alphabet_rocker@0.1.1::proc::install` | `FILE:rubygems::alphabet_rocker@0.1.1::file::default/english-0.7.1.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::alphabet_rocker@0.1.1::proc::install` | `FILE:rubygems::alphabet_rocker@0.1.1::file::default/drb-2.0.4.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::alphabet_rocker@0.1.1::proc::install` | `FILE:rubygems::alphabet_rocker@0.1.1::file::default/digest-3.0.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::alphabet_rocker@0.1.1::proc::install` | `FILE:rubygems::alphabet_rocker@0.1.1::file::default/did_you_mean-1.5.0.gemspec` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:rubygems::alphabet_rocker@0.1.1::proc::install` | `FILE:rubygems::alphabet_rocker@0.1.1::file::socket:[10]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:rubygems::alphabet_rocker@0.1.1::proc::install` | `FILE:rubygems::alphabet_rocker@0.1.1::file::host:[5]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:rubygems::alphabet_rocker@0.1.1::proc::install` | `FILE:rubygems::alphabet_rocker@0.1.1::file::anon_inode:[eventfd]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:rubygems::alphabet_rocker@0.1.1::proc::install` | `FILE:rubygems::alphabet_rocker@0.1.1::file::socket:[6]` |

#### Top-K suspicious edges (rarity-adjusted)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| 0.5002 | `PROC|CONNECT|NET` | `PROC:rubygems::alphabet_rocker@0.1.1::proc::install` | `NET:host::rubygems.org` |
| 0.5002 | `PROC|CONNECT|NET` | `PROC:rubygems::alphabet_rocker@0.1.1::proc::install` | `NET:host::index.rubygems.org` |
| 0.5000 | `PROC|EXEC|CMD` | `PROC:rubygems::alphabet_rocker@0.1.1::proc::install` | `CMD:cmd::rubybin/analyze-ruby.rb--version0.1.1installalphabet_rocker` |
| 0.5000 | `PROC|EXEC|CMD` | `PROC:rubygems::alphabet_rocker@0.1.1::proc::install` | `CMD:cmd::rubybin/geminstall-v0.1.1alphabet_rocker` |
| 0.5000 | `PROC|EXEC|CMD` | `PROC:rubygems::alphabet_rocker@0.1.1::proc::install` | `CMD:cmd::geminstall-v0.1.1alphabet_rocker` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::alphabet_rocker@0.1.1::proc::install` | `FILE:rubygems::alphabet_rocker@0.1.1::file::default/fcntl-1.0.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::alphabet_rocker@0.1.1::proc::install` | `FILE:rubygems::alphabet_rocker@0.1.1::file::default/bigdecimal-3.0.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::alphabet_rocker@0.1.1::proc::install` | `FILE:rubygems::alphabet_rocker@0.1.1::file::default/etc-1.2.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::alphabet_rocker@0.1.1::proc::install` | `FILE:rubygems::alphabet_rocker@0.1.1::file::default/erb-2.2.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::alphabet_rocker@0.1.1::proc::install` | `FILE:rubygems::alphabet_rocker@0.1.1::file::default/english-0.7.1.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::alphabet_rocker@0.1.1::proc::install` | `FILE:rubygems::alphabet_rocker@0.1.1::file::default/drb-2.0.4.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::alphabet_rocker@0.1.1::proc::install` | `FILE:rubygems::alphabet_rocker@0.1.1::file::default/digest-3.0.0.gemspec` |

### Case 3: rubygems::a1330ks-bmi@0.0.1.graph

- **graph**: `artifacts/osp_all/rubygems::a1330ks-bmi@0.0.1.graph.pt`
- **label (y/Level)**: `1`
- **triage rank (lower is more suspicious)**: base `1209` -> rarity `364`

#### Top-K suspicious edges (base)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::a1330ks-bmi@0.0.1::proc::install` | `FILE:rubygems::a1330ks-bmi@0.0.1::file::default/fcntl-1.0.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::a1330ks-bmi@0.0.1::proc::install` | `FILE:rubygems::a1330ks-bmi@0.0.1::file::default/bigdecimal-3.0.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::a1330ks-bmi@0.0.1::proc::install` | `FILE:rubygems::a1330ks-bmi@0.0.1::file::default/etc-1.2.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::a1330ks-bmi@0.0.1::proc::install` | `FILE:rubygems::a1330ks-bmi@0.0.1::file::default/erb-2.2.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::a1330ks-bmi@0.0.1::proc::install` | `FILE:rubygems::a1330ks-bmi@0.0.1::file::default/english-0.7.1.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::a1330ks-bmi@0.0.1::proc::install` | `FILE:rubygems::a1330ks-bmi@0.0.1::file::default/drb-2.0.4.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::a1330ks-bmi@0.0.1::proc::install` | `FILE:rubygems::a1330ks-bmi@0.0.1::file::default/digest-3.0.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::a1330ks-bmi@0.0.1::proc::install` | `FILE:rubygems::a1330ks-bmi@0.0.1::file::default/did_you_mean-1.5.0.gemspec` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:rubygems::a1330ks-bmi@0.0.1::proc::install` | `FILE:rubygems::a1330ks-bmi@0.0.1::file::socket:[11]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:rubygems::a1330ks-bmi@0.0.1::proc::install` | `FILE:rubygems::a1330ks-bmi@0.0.1::file::host:[5]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:rubygems::a1330ks-bmi@0.0.1::proc::install` | `FILE:rubygems::a1330ks-bmi@0.0.1::file::anon_inode:[eventfd]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:rubygems::a1330ks-bmi@0.0.1::proc::install` | `FILE:rubygems::a1330ks-bmi@0.0.1::file::socket:[6]` |

#### Top-K suspicious edges (rarity-adjusted)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| 0.5002 | `PROC|CONNECT|NET` | `PROC:rubygems::a1330ks-bmi@0.0.1::proc::install` | `NET:host::index.rubygems.org` |
| 0.5002 | `PROC|CONNECT|NET` | `PROC:rubygems::a1330ks-bmi@0.0.1::proc::install` | `NET:host::rubygems.org` |
| 0.5000 | `PROC|EXEC|CMD` | `PROC:rubygems::a1330ks-bmi@0.0.1::proc::install` | `CMD:cmd::rubybin/analyze-ruby.rb--version0.0.1install` |
| 0.5000 | `PROC|EXEC|CMD` | `PROC:rubygems::a1330ks-bmi@0.0.1::proc::install` | `CMD:cmd::rubybin/geminstall-v0.0.1` |
| 0.5000 | `PROC|EXEC|CMD` | `PROC:rubygems::a1330ks-bmi@0.0.1::proc::install` | `CMD:cmd::geminstall-v0.0.1` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::a1330ks-bmi@0.0.1::proc::install` | `FILE:rubygems::a1330ks-bmi@0.0.1::file::default/fcntl-1.0.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::a1330ks-bmi@0.0.1::proc::install` | `FILE:rubygems::a1330ks-bmi@0.0.1::file::default/bigdecimal-3.0.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::a1330ks-bmi@0.0.1::proc::install` | `FILE:rubygems::a1330ks-bmi@0.0.1::file::default/etc-1.2.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::a1330ks-bmi@0.0.1::proc::install` | `FILE:rubygems::a1330ks-bmi@0.0.1::file::default/erb-2.2.0.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::a1330ks-bmi@0.0.1::proc::install` | `FILE:rubygems::a1330ks-bmi@0.0.1::file::default/english-0.7.1.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::a1330ks-bmi@0.0.1::proc::install` | `FILE:rubygems::a1330ks-bmi@0.0.1::file::default/drb-2.0.4.gemspec` |
| -0.0000 | `PROC|READ|FILE` | `PROC:rubygems::a1330ks-bmi@0.0.1::proc::install` | `FILE:rubygems::a1330ks-bmi@0.0.1::file::default/digest-3.0.0.gemspec` |
