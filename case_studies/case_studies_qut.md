## Case studies (auto-generated)

Config: k=20 rarity_lambda=0.5 idf_cap=3.0 rarity_etypes=['PROC|CONNECT|NET', 'PROC|EXEC|CMD']

### Case 1: pilolw-0.1.tar.gz.graph

- **graph**: `artifacts/qut_all/pilolw-0.1.tar.gz.graph.pt`
- **label (y/Level)**: `1`
- **triage rank (lower is more suspicious)**: base `1490` -> rarity `14`

#### Top-K suspicious edges (base)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| -0.0000 | `PROC|CONNECT|NET` | `PROC:qut::pilolw-0.1.tar.gz::proc::install` | `NET:port::80` |
| -0.0000 | `PROC|CONNECT|NET` | `PROC:qut::pilolw-0.1.tar.gz::proc::install` | `NET:port::8080` |
| -0.0000 | `PROC|CONNECT|NET` | `PROC:qut::pilolw-0.1.tar.gz::proc::install` | `NET:port::443` |
| -0.0000 | `PROC|WRITE|FILE` | `PROC:qut::pilolw-0.1.tar.gz::proc::install` | `FILE:qut::pilolw-0.1.tar.gz::bucket::ALL_FILES` |
| -0.0000 | `PROC|READ|FILE` | `PROC:qut::pilolw-0.1.tar.gz::proc::install` | `FILE:qut::pilolw-0.1.tar.gz::bucket::ALL_FILES` |
| -0.0000 | `PROC|WRITE|FILE` | `PROC:qut::pilolw-0.1.tar.gz::proc::install` | `FILE:qut::pilolw-0.1.tar.gz::bucket::HOME_DIR` |
| -0.0000 | `PROC|WRITE|FILE` | `PROC:qut::pilolw-0.1.tar.gz::proc::install` | `FILE:qut::pilolw-0.1.tar.gz::bucket::ETC_DIR` |
| -0.0000 | `PROC|WRITE|FILE` | `PROC:qut::pilolw-0.1.tar.gz::proc::install` | `FILE:qut::pilolw-0.1.tar.gz::bucket::OTHER_DIR` |
| -0.0000 | `PROC|WRITE|FILE` | `PROC:qut::pilolw-0.1.tar.gz::proc::install` | `FILE:qut::pilolw-0.1.tar.gz::bucket::TMP_DIR` |
| -0.0000 | `PROC|INVOKE|SYSCALL` | `PROC:qut::pilolw-0.1.tar.gz::proc::install` | `SYSCALL:syscall::mmap` |
| -0.0000 | `PROC|INVOKE|SYSCALL` | `PROC:qut::pilolw-0.1.tar.gz::proc::install` | `SYSCALL:syscall::chmod` |
| -0.0000 | `PROC|INVOKE|SYSCALL` | `PROC:qut::pilolw-0.1.tar.gz::proc::install` | `SYSCALL:syscall::getrandom` |

#### Top-K suspicious edges (rarity-adjusted)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| 1.5000 | `PROC|CONNECT|NET` | `PROC:qut::pilolw-0.1.tar.gz::proc::install` | `NET:port::8080` |
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::pilolw-0.1.tar.gz::proc::install` | `CMD:pattern::7::fstat -> ioctl -> lseek -> no-error -> no-fd` |
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::pilolw-0.1.tar.gz::proc::install` | `CMD:pattern::2::openat -> openat -> openat` |
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::pilolw-0.1.tar.gz::proc::install` | `CMD:pattern::3::fstat -> ioctl -> lseek` |
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::pilolw-0.1.tar.gz::proc::install` | `CMD:pattern::4::read -> read -> close` |
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::pilolw-0.1.tar.gz::proc::install` | `CMD:pattern::5::lseek -> fstat -> read` |
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::pilolw-0.1.tar.gz::proc::install` | `CMD:pattern::6::openat -> openat -> openat -> error=ENOENT -> no-fd` |
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::pilolw-0.1.tar.gz::proc::install` | `CMD:pattern::8::newfstatat -> newfstatat -> newfstatat -> no-error -> no-fd` |
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::pilolw-0.1.tar.gz::proc::install` | `CMD:pattern::9::read -> read -> close -> no-error -> no-fd` |
| 0.5636 | `PROC|CONNECT|NET` | `PROC:qut::pilolw-0.1.tar.gz::proc::install` | `NET:port::80` |
| 0.0938 | `PROC|CONNECT|NET` | `PROC:qut::pilolw-0.1.tar.gz::proc::install` | `NET:port::443` |
| -0.0000 | `PROC|WRITE|FILE` | `PROC:qut::pilolw-0.1.tar.gz::proc::install` | `FILE:qut::pilolw-0.1.tar.gz::bucket::ALL_FILES` |

### Case 2: proggressbar2-0.1.tar.gz.graph

- **graph**: `artifacts/qut_all/proggressbar2-0.1.tar.gz.graph.pt`
- **label (y/Level)**: `1`
- **triage rank (lower is more suspicious)**: base `1486` -> rarity `22`

#### Top-K suspicious edges (base)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| -0.0000 | `PROC|CONNECT|NET` | `PROC:qut::proggressbar2-0.1.tar.gz::proc::install` | `NET:port::80` |
| -0.0000 | `PROC|CONNECT|NET` | `PROC:qut::proggressbar2-0.1.tar.gz::proc::install` | `NET:port::443` |
| -0.0000 | `PROC|CONNECT|NET` | `PROC:qut::proggressbar2-0.1.tar.gz::proc::install` | `NET:port::8080` |
| -0.0000 | `PROC|WRITE|FILE` | `PROC:qut::proggressbar2-0.1.tar.gz::proc::install` | `FILE:qut::proggressbar2-0.1.tar.gz::bucket::ALL_FILES` |
| -0.0000 | `PROC|READ|FILE` | `PROC:qut::proggressbar2-0.1.tar.gz::proc::install` | `FILE:qut::proggressbar2-0.1.tar.gz::bucket::ALL_FILES` |
| -0.0000 | `PROC|WRITE|FILE` | `PROC:qut::proggressbar2-0.1.tar.gz::proc::install` | `FILE:qut::proggressbar2-0.1.tar.gz::bucket::OTHER_DIR` |
| -0.0000 | `PROC|WRITE|FILE` | `PROC:qut::proggressbar2-0.1.tar.gz::proc::install` | `FILE:qut::proggressbar2-0.1.tar.gz::bucket::ETC_DIR` |
| -0.0000 | `PROC|WRITE|FILE` | `PROC:qut::proggressbar2-0.1.tar.gz::proc::install` | `FILE:qut::proggressbar2-0.1.tar.gz::bucket::HOME_DIR` |
| -0.0000 | `PROC|WRITE|FILE` | `PROC:qut::proggressbar2-0.1.tar.gz::proc::install` | `FILE:qut::proggressbar2-0.1.tar.gz::bucket::TMP_DIR` |
| -0.0000 | `PROC|WRITE|FILE` | `PROC:qut::proggressbar2-0.1.tar.gz::proc::install` | `FILE:qut::proggressbar2-0.1.tar.gz::bucket::ROOT_DIR` |
| -0.0000 | `PROC|INVOKE|SYSCALL` | `PROC:qut::proggressbar2-0.1.tar.gz::proc::install` | `SYSCALL:syscall::munmap` |
| -0.0000 | `PROC|INVOKE|SYSCALL` | `PROC:qut::proggressbar2-0.1.tar.gz::proc::install` | `SYSCALL:syscall::fsync` |

#### Top-K suspicious edges (rarity-adjusted)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| 1.5000 | `PROC|CONNECT|NET` | `PROC:qut::proggressbar2-0.1.tar.gz::proc::install` | `NET:port::8080` |
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::proggressbar2-0.1.tar.gz::proc::install` | `CMD:pattern::1::lseek -> lseek -> lseek` |
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::proggressbar2-0.1.tar.gz::proc::install` | `CMD:pattern::3::newfstatat -> newfstatat -> newfstatat` |
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::proggressbar2-0.1.tar.gz::proc::install` | `CMD:pattern::4::openat -> fstat -> ioctl` |
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::proggressbar2-0.1.tar.gz::proc::install` | `CMD:pattern::6::lseek -> lseek -> lseek -> no-error -> no-fd` |
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::proggressbar2-0.1.tar.gz::proc::install` | `CMD:pattern::8::openat -> fstat -> ioctl -> no-error -> no-fd` |
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::proggressbar2-0.1.tar.gz::proc::install` | `CMD:pattern::10::ioctl -> lseek -> lseek -> error=ENOTTY -> no-fd` |
| 1.4640 | `PROC|EXEC|CMD` | `PROC:qut::proggressbar2-0.1.tar.gz::proc::install` | `CMD:pattern::9::fstat -> ioctl -> lseek -> no-error -> no-fd` |
| 1.2939 | `PROC|EXEC|CMD` | `PROC:qut::proggressbar2-0.1.tar.gz::proc::install` | `CMD:pattern::2::newfstatat -> openat -> fstat` |
| 0.5636 | `PROC|CONNECT|NET` | `PROC:qut::proggressbar2-0.1.tar.gz::proc::install` | `NET:port::80` |
| 0.0938 | `PROC|CONNECT|NET` | `PROC:qut::proggressbar2-0.1.tar.gz::proc::install` | `NET:port::443` |
| -0.0000 | `PROC|WRITE|FILE` | `PROC:qut::proggressbar2-0.1.tar.gz::proc::install` | `FILE:qut::proggressbar2-0.1.tar.gz::bucket::ALL_FILES` |

### Case 3: progessbar2-0.1.tar.gz.graph

- **graph**: `artifacts/qut_all/progessbar2-0.1.tar.gz.graph.pt`
- **label (y/Level)**: `1`
- **triage rank (lower is more suspicious)**: base `1471` -> rarity `42`

#### Top-K suspicious edges (base)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| -0.0000 | `PROC|CONNECT|NET` | `PROC:qut::progessbar2-0.1.tar.gz::proc::install` | `NET:port::80` |
| -0.0000 | `PROC|CONNECT|NET` | `PROC:qut::progessbar2-0.1.tar.gz::proc::install` | `NET:port::8080` |
| -0.0000 | `PROC|CONNECT|NET` | `PROC:qut::progessbar2-0.1.tar.gz::proc::install` | `NET:port::443` |
| -0.0000 | `PROC|WRITE|FILE` | `PROC:qut::progessbar2-0.1.tar.gz::proc::install` | `FILE:qut::progessbar2-0.1.tar.gz::bucket::ALL_FILES` |
| -0.0000 | `PROC|READ|FILE` | `PROC:qut::progessbar2-0.1.tar.gz::proc::install` | `FILE:qut::progessbar2-0.1.tar.gz::bucket::ALL_FILES` |
| -0.0000 | `PROC|WRITE|FILE` | `PROC:qut::progessbar2-0.1.tar.gz::proc::install` | `FILE:qut::progessbar2-0.1.tar.gz::bucket::HOME_DIR` |
| -0.0000 | `PROC|WRITE|FILE` | `PROC:qut::progessbar2-0.1.tar.gz::proc::install` | `FILE:qut::progessbar2-0.1.tar.gz::bucket::OTHER_DIR` |
| -0.0000 | `PROC|WRITE|FILE` | `PROC:qut::progessbar2-0.1.tar.gz::proc::install` | `FILE:qut::progessbar2-0.1.tar.gz::bucket::ETC_DIR` |
| -0.0000 | `PROC|WRITE|FILE` | `PROC:qut::progessbar2-0.1.tar.gz::proc::install` | `FILE:qut::progessbar2-0.1.tar.gz::bucket::TMP_DIR` |
| -0.0000 | `PROC|WRITE|FILE` | `PROC:qut::progessbar2-0.1.tar.gz::proc::install` | `FILE:qut::progessbar2-0.1.tar.gz::bucket::ROOT_DIR` |
| -0.0000 | `PROC|INVOKE|SYSCALL` | `PROC:qut::progessbar2-0.1.tar.gz::proc::install` | `SYSCALL:syscall::sigaltstack` |
| -0.0000 | `PROC|INVOKE|SYSCALL` | `PROC:qut::progessbar2-0.1.tar.gz::proc::install` | `SYSCALL:syscall::mmap` |

#### Top-K suspicious edges (rarity-adjusted)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| 1.5000 | `PROC|CONNECT|NET` | `PROC:qut::progessbar2-0.1.tar.gz::proc::install` | `NET:port::8080` |
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::progessbar2-0.1.tar.gz::proc::install` | `CMD:pattern::6::lseek -> lseek -> lseek -> no-error -> no-fd` |
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::progessbar2-0.1.tar.gz::proc::install` | `CMD:pattern::1::lseek -> lseek -> lseek` |
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::progessbar2-0.1.tar.gz::proc::install` | `CMD:pattern::4::openat -> fstat -> ioctl` |
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::progessbar2-0.1.tar.gz::proc::install` | `CMD:pattern::8::openat -> fstat -> ioctl -> no-error -> no-fd` |
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::progessbar2-0.1.tar.gz::proc::install` | `CMD:pattern::10::ioctl -> lseek -> lseek -> error=ENOTTY -> no-fd` |
| 1.4640 | `PROC|EXEC|CMD` | `PROC:qut::progessbar2-0.1.tar.gz::proc::install` | `CMD:pattern::9::fstat -> ioctl -> lseek -> no-error -> no-fd` |
| 0.8872 | `PROC|EXEC|CMD` | `PROC:qut::progessbar2-0.1.tar.gz::proc::install` | `CMD:pattern::2::newfstatat -> newfstatat -> newfstatat` |
| 0.8624 | `PROC|EXEC|CMD` | `PROC:qut::progessbar2-0.1.tar.gz::proc::install` | `CMD:pattern::7::newfstatat -> newfstatat -> newfstatat -> no-error -> no-fd` |
| 0.5636 | `PROC|CONNECT|NET` | `PROC:qut::progessbar2-0.1.tar.gz::proc::install` | `NET:port::80` |
| 0.0938 | `PROC|CONNECT|NET` | `PROC:qut::progessbar2-0.1.tar.gz::proc::install` | `NET:port::443` |
| -0.0000 | `PROC|WRITE|FILE` | `PROC:qut::progessbar2-0.1.tar.gz::proc::install` | `FILE:qut::progessbar2-0.1.tar.gz::bucket::ALL_FILES` |
