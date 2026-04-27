## Case studies (auto-generated)

Config: k=20 rarity_lambda=0.5 idf_cap=3.0 rarity_etypes=['PROC|CONNECT|NET', 'PROC|EXEC|CMD']
Included packages: ['10Cent10', '10Cent11']

### Case 1: 10Cent10-999.0.4.tar.gz.graph

- **graph**: `artifacts/qut_a/10Cent10-999.0.4.tar.gz.graph.pt`
- **label (y/Level)**: `1`
- **triage rank (lower is more suspicious)**: base `288` -> rarity `8`

#### Top-K suspicious edges (base)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| -0.0000 | `PROC|EXEC|CMD` | `PROC:qut::10Cent10-999.0.4.tar.gz::proc::install` | `CMD:pattern::10::read -> read -> close -> no-error -> no-fd` |
| -0.0000 | `PROC|EXEC|CMD` | `PROC:qut::10Cent10-999.0.4.tar.gz::proc::install` | `CMD:pattern::9::ioctl -> lseek -> lseek -> error=ENOTTY -> no-fd` |
| -0.0000 | `PROC|EXEC|CMD` | `PROC:qut::10Cent10-999.0.4.tar.gz::proc::install` | `CMD:pattern::8::openat -> fstat -> ioctl -> no-error -> no-fd` |
| -0.0000 | `PROC|EXEC|CMD` | `PROC:qut::10Cent10-999.0.4.tar.gz::proc::install` | `CMD:pattern::6::newfstatat -> newfstatat -> newfstatat -> no-error -> no-fd` |
| -0.0000 | `PROC|EXEC|CMD` | `PROC:qut::10Cent10-999.0.4.tar.gz::proc::install` | `CMD:pattern::4::newfstatat -> newfstatat -> openat` |
| -0.0000 | `PROC|EXEC|CMD` | `PROC:qut::10Cent10-999.0.4.tar.gz::proc::install` | `CMD:pattern::3::openat -> fstat -> ioctl` |
| -0.0000 | `PROC|EXEC|CMD` | `PROC:qut::10Cent10-999.0.4.tar.gz::proc::install` | `CMD:pattern::7::fstat -> ioctl -> lseek -> no-error -> no-fd` |
| -0.0000 | `PROC|EXEC|CMD` | `PROC:qut::10Cent10-999.0.4.tar.gz::proc::install` | `CMD:pattern::5::newfstatat -> openat -> fstat` |
| -0.0000 | `PROC|INVOKE|SYSCALL` | `PROC:qut::10Cent10-999.0.4.tar.gz::proc::install` | `SYSCALL:syscall::getpid` |
| -0.0000 | `PROC|INVOKE|SYSCALL` | `PROC:qut::10Cent10-999.0.4.tar.gz::proc::install` | `SYSCALL:syscall::getdents64` |
| -0.0000 | `PROC|INVOKE|SYSCALL` | `PROC:qut::10Cent10-999.0.4.tar.gz::proc::install` | `SYSCALL:syscall::epoll_create1` |
| -0.0000 | `PROC|INVOKE|SYSCALL` | `PROC:qut::10Cent10-999.0.4.tar.gz::proc::install` | `SYSCALL:syscall::uname` |

#### Top-K suspicious edges (rarity-adjusted)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::10Cent10-999.0.4.tar.gz::proc::install` | `CMD:pattern::2::fstat -> ioctl -> lseek` |
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::10Cent10-999.0.4.tar.gz::proc::install` | `CMD:pattern::10::read -> read -> close -> no-error -> no-fd` |
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::10Cent10-999.0.4.tar.gz::proc::install` | `CMD:pattern::9::ioctl -> lseek -> lseek -> error=ENOTTY -> no-fd` |
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::10Cent10-999.0.4.tar.gz::proc::install` | `CMD:pattern::8::openat -> fstat -> ioctl -> no-error -> no-fd` |
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::10Cent10-999.0.4.tar.gz::proc::install` | `CMD:pattern::3::openat -> fstat -> ioctl` |
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::10Cent10-999.0.4.tar.gz::proc::install` | `CMD:pattern::7::fstat -> ioctl -> lseek -> no-error -> no-fd` |
| 1.5000 | `PROC|CONNECT|NET` | `PROC:qut::10Cent10-999.0.4.tar.gz::proc::install` | `NET:ip::62` |
| 1.5000 | `PROC|CONNECT|NET` | `PROC:qut::10Cent10-999.0.4.tar.gz::proc::install` | `NET:port::21` |
| 1.4388 | `PROC|EXEC|CMD` | `PROC:qut::10Cent10-999.0.4.tar.gz::proc::install` | `CMD:pattern::5::newfstatat -> openat -> fstat` |
| 1.0234 | `PROC|EXEC|CMD` | `PROC:qut::10Cent10-999.0.4.tar.gz::proc::install` | `CMD:pattern::4::newfstatat -> newfstatat -> openat` |
| -0.0000 | `PROC|INVOKE|SYSCALL` | `PROC:qut::10Cent10-999.0.4.tar.gz::proc::install` | `SYSCALL:syscall::getpid` |
| -0.0000 | `PROC|INVOKE|SYSCALL` | `PROC:qut::10Cent10-999.0.4.tar.gz::proc::install` | `SYSCALL:syscall::getdents64` |

### Case 2: 10Cent11-999.0.4.tar.gz.graph

- **graph**: `artifacts/qut_a/10Cent11-999.0.4.tar.gz.graph.pt`
- **label (y/Level)**: `1`
- **triage rank (lower is more suspicious)**: base `144` -> rarity `727`

#### Top-K suspicious edges (base)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| -0.0000 | `PROC|EXEC|CMD` | `PROC:qut::10Cent11-999.0.4.tar.gz::proc::install` | `CMD:pattern::10::fcntl -> fstat -> fcntl -> no-error -> no-fd` |
| -0.0000 | `PROC|EXEC|CMD` | `PROC:qut::10Cent11-999.0.4.tar.gz::proc::install` | `CMD:pattern::9::ioctl -> ioctl -> ioctl -> no-error -> no-fd` |
| -0.0000 | `PROC|EXEC|CMD` | `PROC:qut::10Cent11-999.0.4.tar.gz::proc::install` | `CMD:pattern::8::newfstatat -> openat -> fstat -> no-error -> no-fd` |
| -0.0000 | `PROC|EXEC|CMD` | `PROC:qut::10Cent11-999.0.4.tar.gz::proc::install` | `CMD:pattern::3::newfstatat -> openat -> fstat` |
| -0.0000 | `PROC|EXEC|CMD` | `PROC:qut::10Cent11-999.0.4.tar.gz::proc::install` | `CMD:pattern::5::newfstatat -> newfstatat -> openat` |
| -0.0000 | `PROC|EXEC|CMD` | `PROC:qut::10Cent11-999.0.4.tar.gz::proc::install` | `CMD:pattern::1::newfstatat -> newfstatat -> newfstatat` |
| -0.0000 | `PROC|EXEC|CMD` | `PROC:qut::10Cent11-999.0.4.tar.gz::proc::install` | `CMD:pattern::4::ioctl -> ioctl -> ioctl` |
| -0.0000 | `PROC|EXEC|CMD` | `PROC:qut::10Cent11-999.0.4.tar.gz::proc::install` | `CMD:pattern::6::newfstatat -> newfstatat -> newfstatat -> no-error -> no-fd` |
| -0.0000 | `PROC|INVOKE|SYSCALL` | `PROC:qut::10Cent11-999.0.4.tar.gz::proc::install` | `SYSCALL:syscall::mprotect` |
| -0.0000 | `PROC|INVOKE|SYSCALL` | `PROC:qut::10Cent11-999.0.4.tar.gz::proc::install` | `SYSCALL:syscall::unlinkat` |
| -0.0000 | `PROC|INVOKE|SYSCALL` | `PROC:qut::10Cent11-999.0.4.tar.gz::proc::install` | `SYSCALL:syscall::rmdir` |
| -0.0000 | `PROC|INVOKE|SYSCALL` | `PROC:qut::10Cent11-999.0.4.tar.gz::proc::install` | `SYSCALL:syscall::socket` |

#### Top-K suspicious edges (rarity-adjusted)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::10Cent11-999.0.4.tar.gz::proc::install` | `CMD:pattern::5::newfstatat -> newfstatat -> openat` |
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::10Cent11-999.0.4.tar.gz::proc::install` | `CMD:pattern::4::ioctl -> ioctl -> ioctl` |
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::10Cent11-999.0.4.tar.gz::proc::install` | `CMD:pattern::10::fcntl -> fstat -> fcntl -> no-error -> no-fd` |
| 1.5000 | `PROC|EXEC|CMD` | `PROC:qut::10Cent11-999.0.4.tar.gz::proc::install` | `CMD:pattern::9::ioctl -> ioctl -> ioctl -> no-error -> no-fd` |
| 1.5000 | `PROC|CONNECT|NET` | `PROC:qut::10Cent11-999.0.4.tar.gz::proc::install` | `NET:ip::51` |
| 1.5000 | `PROC|CONNECT|NET` | `PROC:qut::10Cent11-999.0.4.tar.gz::proc::install` | `NET:port::18` |
| 0.6468 | `PROC|EXEC|CMD` | `PROC:qut::10Cent11-999.0.4.tar.gz::proc::install` | `CMD:pattern::3::newfstatat -> openat -> fstat` |
| 0.6420 | `PROC|EXEC|CMD` | `PROC:qut::10Cent11-999.0.4.tar.gz::proc::install` | `CMD:pattern::8::newfstatat -> openat -> fstat -> no-error -> no-fd` |
| 0.4348 | `PROC|EXEC|CMD` | `PROC:qut::10Cent11-999.0.4.tar.gz::proc::install` | `CMD:pattern::7::read -> read -> read -> no-error -> no-fd` |
| 0.4085 | `PROC|EXEC|CMD` | `PROC:qut::10Cent11-999.0.4.tar.gz::proc::install` | `CMD:pattern::2::read -> read -> read` |
| -0.0000 | `PROC|INVOKE|SYSCALL` | `PROC:qut::10Cent11-999.0.4.tar.gz::proc::install` | `SYSCALL:syscall::mprotect` |
| -0.0000 | `PROC|INVOKE|SYSCALL` | `PROC:qut::10Cent11-999.0.4.tar.gz::proc::install` | `SYSCALL:syscall::unlinkat` |

### Case 3: pilolw-0.1.tar.gz.graph

- **graph**: `artifacts/qut_all/pilolw-0.1.tar.gz.graph.pt`
- **label (y/Level)**: `1`
- **triage rank (lower is more suspicious)**: base `1492` -> rarity `15`

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
