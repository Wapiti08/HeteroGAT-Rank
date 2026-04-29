## Case studies (auto-generated)

Config: k=20 rarity_lambda=0.5 idf_cap=3.0 rarity_etypes=['PROC|CONNECT|NET', 'PROC|EXEC|CMD']
Included packages: ['capmonstercloudclent', 'capmonstercloudclinent', 'capmonstercloudclieent', 'capmonstercloudclinet', 'capmonstercloudclouidclient', 'capmonsstercloudclient', 'ligitgays', 'bettercolors']

### Case 1: pypi::capmonstercloudclent@1.0.0.graph

- **graph**: `artifacts/osp_all/pypi::capmonstercloudclent@1.0.0.graph.pt`
- **label (y/Level)**: `1`
- **triage rank (lower is more suspicious)**: base `368` -> rarity `716`

#### Top-K suspicious edges (base)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclent@1.0.0::file::command/__init__.abi3.so` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclent@1.0.0::file::__pycache__/_functools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclent@1.0.0::file::command/install.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclent@1.0.0::file::__pycache__/_itertools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclent@1.0.0::file::distutils/command` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclent@1.0.0::file::distutils/cmd.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclent@1.0.0::file::distutils/archive_util.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclent@1.0.0::file::__pycache__/util.cpython-310.pyc` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclent@1.0.0::file::pipe:[1]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclent@1.0.0::file::dev/null` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclent@1.0.0::file::host:[5]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclent@1.0.0::file::host:[6]` |

#### Top-K suspicious edges (rarity-adjusted)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| 0.5001 | `PROC|EXEC|CMD` | `PROC:pypi::capmonstercloudclent@1.0.0::proc::install` | `CMD:cmd::bin/python3-mpipinstall--precapmonstercloudclent==1.0.0` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclent@1.0.0::file::command/__init__.abi3.so` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclent@1.0.0::file::__pycache__/_functools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclent@1.0.0::file::command/install.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclent@1.0.0::file::__pycache__/_itertools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclent@1.0.0::file::distutils/command` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclent@1.0.0::file::distutils/cmd.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclent@1.0.0::file::distutils/archive_util.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclent@1.0.0::file::__pycache__/util.cpython-310.pyc` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclent@1.0.0::file::pipe:[1]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclent@1.0.0::file::dev/null` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclent@1.0.0::file::host:[5]` |

### Case 2: pypi::capmonstercloudclinent@1.0.0.graph

- **graph**: `artifacts/osp_all/pypi::capmonstercloudclinent@1.0.0.graph.pt`
- **label (y/Level)**: `1`
- **triage rank (lower is more suspicious)**: base `369` -> rarity `717`

#### Top-K suspicious edges (base)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinent@1.0.0::file::command/__init__.abi3.so` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinent@1.0.0::file::__pycache__/_functools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinent@1.0.0::file::command/install.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinent@1.0.0::file::__pycache__/_itertools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinent@1.0.0::file::distutils/command` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinent@1.0.0::file::distutils/cmd.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinent@1.0.0::file::distutils/archive_util.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinent@1.0.0::file::__pycache__/util.cpython-310.pyc` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclinent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinent@1.0.0::file::pipe:[3]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclinent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinent@1.0.0::file::pipe:[1]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclinent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinent@1.0.0::file::pipe:[10]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclinent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinent@1.0.0::file::host:[6]` |

#### Top-K suspicious edges (rarity-adjusted)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| 0.5001 | `PROC|EXEC|CMD` | `PROC:pypi::capmonstercloudclinent@1.0.0::proc::install` | `CMD:cmd::bin/python3-mpipinstall--precapmonstercloudclinent==1.0.0` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinent@1.0.0::file::command/__init__.abi3.so` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinent@1.0.0::file::__pycache__/_functools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinent@1.0.0::file::command/install.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinent@1.0.0::file::__pycache__/_itertools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinent@1.0.0::file::distutils/command` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinent@1.0.0::file::distutils/cmd.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinent@1.0.0::file::distutils/archive_util.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinent@1.0.0::file::__pycache__/util.cpython-310.pyc` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclinent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinent@1.0.0::file::pipe:[3]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclinent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinent@1.0.0::file::pipe:[1]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclinent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinent@1.0.0::file::pipe:[10]` |

### Case 3: pypi::capmonstercloudclieent@1.0.0.graph

- **graph**: `artifacts/osp_all/pypi::capmonstercloudclieent@1.0.0.graph.pt`
- **label (y/Level)**: `1`
- **triage rank (lower is more suspicious)**: base `370` -> rarity `718`

#### Top-K suspicious edges (base)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclieent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclieent@1.0.0::file::command/__init__.abi3.so` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclieent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclieent@1.0.0::file::__pycache__/_functools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclieent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclieent@1.0.0::file::command/install.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclieent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclieent@1.0.0::file::__pycache__/_itertools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclieent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclieent@1.0.0::file::distutils/command` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclieent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclieent@1.0.0::file::distutils/cmd.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclieent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclieent@1.0.0::file::distutils/archive_util.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclieent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclieent@1.0.0::file::__pycache__/util.cpython-310.pyc` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclieent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclieent@1.0.0::file::pipe:[3]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclieent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclieent@1.0.0::file::pipe:[1]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclieent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclieent@1.0.0::file::pipe:[10]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclieent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclieent@1.0.0::file::host:[6]` |

#### Top-K suspicious edges (rarity-adjusted)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| 0.5001 | `PROC|EXEC|CMD` | `PROC:pypi::capmonstercloudclieent@1.0.0::proc::install` | `CMD:cmd::bin/python3-mpipinstall--precapmonstercloudclieent==1.0.0` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclieent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclieent@1.0.0::file::command/__init__.abi3.so` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclieent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclieent@1.0.0::file::__pycache__/_functools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclieent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclieent@1.0.0::file::command/install.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclieent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclieent@1.0.0::file::__pycache__/_itertools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclieent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclieent@1.0.0::file::distutils/command` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclieent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclieent@1.0.0::file::distutils/cmd.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclieent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclieent@1.0.0::file::distutils/archive_util.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclieent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclieent@1.0.0::file::__pycache__/util.cpython-310.pyc` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclieent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclieent@1.0.0::file::pipe:[3]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclieent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclieent@1.0.0::file::pipe:[1]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclieent@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclieent@1.0.0::file::pipe:[10]` |

### Case 4: pypi::capmonstercloudclinet@1.0.0.graph

- **graph**: `artifacts/osp_all/pypi::capmonstercloudclinet@1.0.0.graph.pt`
- **label (y/Level)**: `1`
- **triage rank (lower is more suspicious)**: base `371` -> rarity `719`

#### Top-K suspicious edges (base)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinet@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinet@1.0.0::file::command/__init__.abi3.so` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinet@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinet@1.0.0::file::__pycache__/_functools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinet@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinet@1.0.0::file::command/install.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinet@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinet@1.0.0::file::__pycache__/_itertools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinet@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinet@1.0.0::file::distutils/command` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinet@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinet@1.0.0::file::distutils/cmd.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinet@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinet@1.0.0::file::distutils/archive_util.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinet@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinet@1.0.0::file::__pycache__/util.cpython-310.pyc` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclinet@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinet@1.0.0::file::pipe:[3]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclinet@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinet@1.0.0::file::pipe:[1]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclinet@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinet@1.0.0::file::pipe:[10]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclinet@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinet@1.0.0::file::host:[6]` |

#### Top-K suspicious edges (rarity-adjusted)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| 0.5001 | `PROC|EXEC|CMD` | `PROC:pypi::capmonstercloudclinet@1.0.0::proc::install` | `CMD:cmd::bin/python3-mpipinstall--precapmonstercloudclinet==1.0.0` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinet@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinet@1.0.0::file::command/__init__.abi3.so` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinet@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinet@1.0.0::file::__pycache__/_functools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinet@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinet@1.0.0::file::command/install.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinet@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinet@1.0.0::file::__pycache__/_itertools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinet@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinet@1.0.0::file::distutils/command` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinet@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinet@1.0.0::file::distutils/cmd.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinet@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinet@1.0.0::file::distutils/archive_util.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclinet@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinet@1.0.0::file::__pycache__/util.cpython-310.pyc` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclinet@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinet@1.0.0::file::pipe:[3]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclinet@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinet@1.0.0::file::pipe:[1]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclinet@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclinet@1.0.0::file::pipe:[10]` |

### Case 5: pypi::capmonstercloudclouidclient@1.0.0.graph

- **graph**: `artifacts/osp_all/pypi::capmonstercloudclouidclient@1.0.0.graph.pt`
- **label (y/Level)**: `1`
- **triage rank (lower is more suspicious)**: base `372` -> rarity `720`

#### Top-K suspicious edges (base)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclouidclient@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclouidclient@1.0.0::file::command/__init__.abi3.so` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclouidclient@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclouidclient@1.0.0::file::__pycache__/_functools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclouidclient@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclouidclient@1.0.0::file::command/install.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclouidclient@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclouidclient@1.0.0::file::__pycache__/_itertools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclouidclient@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclouidclient@1.0.0::file::distutils/command` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclouidclient@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclouidclient@1.0.0::file::distutils/cmd.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclouidclient@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclouidclient@1.0.0::file::distutils/archive_util.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclouidclient@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclouidclient@1.0.0::file::__pycache__/util.cpython-310.pyc` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclouidclient@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclouidclient@1.0.0::file::pipe:[3]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclouidclient@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclouidclient@1.0.0::file::pipe:[1]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclouidclient@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclouidclient@1.0.0::file::pipe:[10]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclouidclient@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclouidclient@1.0.0::file::host:[6]` |

#### Top-K suspicious edges (rarity-adjusted)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| 0.5001 | `PROC|EXEC|CMD` | `PROC:pypi::capmonstercloudclouidclient@1.0.0::proc::install` | `CMD:cmd::bin/python3-mpipinstall--precapmonstercloudclouidclient==1.0.0` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclouidclient@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclouidclient@1.0.0::file::command/__init__.abi3.so` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclouidclient@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclouidclient@1.0.0::file::__pycache__/_functools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclouidclient@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclouidclient@1.0.0::file::command/install.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclouidclient@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclouidclient@1.0.0::file::__pycache__/_itertools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclouidclient@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclouidclient@1.0.0::file::distutils/command` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclouidclient@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclouidclient@1.0.0::file::distutils/cmd.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclouidclient@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclouidclient@1.0.0::file::distutils/archive_util.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonstercloudclouidclient@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclouidclient@1.0.0::file::__pycache__/util.cpython-310.pyc` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclouidclient@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclouidclient@1.0.0::file::pipe:[3]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclouidclient@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclouidclient@1.0.0::file::pipe:[1]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonstercloudclouidclient@1.0.0::proc::install` | `FILE:pypi::capmonstercloudclouidclient@1.0.0::file::pipe:[10]` |

### Case 6: pypi::capmonsstercloudclient@1.0.0.graph

- **graph**: `artifacts/osp_all/pypi::capmonsstercloudclient@1.0.0.graph.pt`
- **label (y/Level)**: `1`
- **triage rank (lower is more suspicious)**: base `373` -> rarity `721`

#### Top-K suspicious edges (base)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonsstercloudclient@1.0.0::proc::install` | `FILE:pypi::capmonsstercloudclient@1.0.0::file::command/__init__.abi3.so` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonsstercloudclient@1.0.0::proc::install` | `FILE:pypi::capmonsstercloudclient@1.0.0::file::__pycache__/_functools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonsstercloudclient@1.0.0::proc::install` | `FILE:pypi::capmonsstercloudclient@1.0.0::file::command/install.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonsstercloudclient@1.0.0::proc::install` | `FILE:pypi::capmonsstercloudclient@1.0.0::file::__pycache__/_itertools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonsstercloudclient@1.0.0::proc::install` | `FILE:pypi::capmonsstercloudclient@1.0.0::file::distutils/command` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonsstercloudclient@1.0.0::proc::install` | `FILE:pypi::capmonsstercloudclient@1.0.0::file::distutils/cmd.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonsstercloudclient@1.0.0::proc::install` | `FILE:pypi::capmonsstercloudclient@1.0.0::file::distutils/archive_util.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonsstercloudclient@1.0.0::proc::install` | `FILE:pypi::capmonsstercloudclient@1.0.0::file::__pycache__/util.cpython-310.pyc` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonsstercloudclient@1.0.0::proc::install` | `FILE:pypi::capmonsstercloudclient@1.0.0::file::pipe:[3]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonsstercloudclient@1.0.0::proc::install` | `FILE:pypi::capmonsstercloudclient@1.0.0::file::pipe:[1]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonsstercloudclient@1.0.0::proc::install` | `FILE:pypi::capmonsstercloudclient@1.0.0::file::pipe:[10]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonsstercloudclient@1.0.0::proc::install` | `FILE:pypi::capmonsstercloudclient@1.0.0::file::host:[6]` |

#### Top-K suspicious edges (rarity-adjusted)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| 0.5001 | `PROC|EXEC|CMD` | `PROC:pypi::capmonsstercloudclient@1.0.0::proc::install` | `CMD:cmd::bin/python3-mpipinstall--precapmonsstercloudclient==1.0.0` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonsstercloudclient@1.0.0::proc::install` | `FILE:pypi::capmonsstercloudclient@1.0.0::file::command/__init__.abi3.so` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonsstercloudclient@1.0.0::proc::install` | `FILE:pypi::capmonsstercloudclient@1.0.0::file::__pycache__/_functools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonsstercloudclient@1.0.0::proc::install` | `FILE:pypi::capmonsstercloudclient@1.0.0::file::command/install.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonsstercloudclient@1.0.0::proc::install` | `FILE:pypi::capmonsstercloudclient@1.0.0::file::__pycache__/_itertools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonsstercloudclient@1.0.0::proc::install` | `FILE:pypi::capmonsstercloudclient@1.0.0::file::distutils/command` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonsstercloudclient@1.0.0::proc::install` | `FILE:pypi::capmonsstercloudclient@1.0.0::file::distutils/cmd.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonsstercloudclient@1.0.0::proc::install` | `FILE:pypi::capmonsstercloudclient@1.0.0::file::distutils/archive_util.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::capmonsstercloudclient@1.0.0::proc::install` | `FILE:pypi::capmonsstercloudclient@1.0.0::file::__pycache__/util.cpython-310.pyc` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonsstercloudclient@1.0.0::proc::install` | `FILE:pypi::capmonsstercloudclient@1.0.0::file::pipe:[3]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonsstercloudclient@1.0.0::proc::install` | `FILE:pypi::capmonsstercloudclient@1.0.0::file::pipe:[1]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::capmonsstercloudclient@1.0.0::proc::install` | `FILE:pypi::capmonsstercloudclient@1.0.0::file::pipe:[10]` |

### Case 7: pypi::ligitgays@1.0.graph

- **graph**: `artifacts/osp_all/pypi::ligitgays@1.0.graph.pt`
- **label (y/Level)**: `1`
- **triage rank (lower is more suspicious)**: base `374` -> rarity `722`

#### Top-K suspicious edges (base)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::ligitgays@1.0::proc::install` | `FILE:pypi::ligitgays@1.0::file::command/__init__.abi3.so` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::ligitgays@1.0::proc::install` | `FILE:pypi::ligitgays@1.0::file::__pycache__/_functools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::ligitgays@1.0::proc::install` | `FILE:pypi::ligitgays@1.0::file::command/install.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::ligitgays@1.0::proc::install` | `FILE:pypi::ligitgays@1.0::file::__pycache__/_itertools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::ligitgays@1.0::proc::install` | `FILE:pypi::ligitgays@1.0::file::distutils/command` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::ligitgays@1.0::proc::install` | `FILE:pypi::ligitgays@1.0::file::distutils/cmd.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::ligitgays@1.0::proc::install` | `FILE:pypi::ligitgays@1.0::file::distutils/archive_util.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::ligitgays@1.0::proc::install` | `FILE:pypi::ligitgays@1.0::file::__pycache__/util.cpython-310.pyc` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::ligitgays@1.0::proc::install` | `FILE:pypi::ligitgays@1.0::file::pipe:[3]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::ligitgays@1.0::proc::install` | `FILE:pypi::ligitgays@1.0::file::pipe:[1]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::ligitgays@1.0::proc::install` | `FILE:pypi::ligitgays@1.0::file::pipe:[10]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::ligitgays@1.0::proc::install` | `FILE:pypi::ligitgays@1.0::file::host:[6]` |

#### Top-K suspicious edges (rarity-adjusted)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| 0.5001 | `PROC|EXEC|CMD` | `PROC:pypi::ligitgays@1.0::proc::install` | `CMD:cmd::bin/python3-mpipinstall--preligitgays==1.0` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::ligitgays@1.0::proc::install` | `FILE:pypi::ligitgays@1.0::file::command/__init__.abi3.so` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::ligitgays@1.0::proc::install` | `FILE:pypi::ligitgays@1.0::file::__pycache__/_functools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::ligitgays@1.0::proc::install` | `FILE:pypi::ligitgays@1.0::file::command/install.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::ligitgays@1.0::proc::install` | `FILE:pypi::ligitgays@1.0::file::__pycache__/_itertools.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::ligitgays@1.0::proc::install` | `FILE:pypi::ligitgays@1.0::file::distutils/command` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::ligitgays@1.0::proc::install` | `FILE:pypi::ligitgays@1.0::file::distutils/cmd.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::ligitgays@1.0::proc::install` | `FILE:pypi::ligitgays@1.0::file::distutils/archive_util.py` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::ligitgays@1.0::proc::install` | `FILE:pypi::ligitgays@1.0::file::__pycache__/util.cpython-310.pyc` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::ligitgays@1.0::proc::install` | `FILE:pypi::ligitgays@1.0::file::pipe:[3]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::ligitgays@1.0::proc::install` | `FILE:pypi::ligitgays@1.0::file::pipe:[1]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::ligitgays@1.0::proc::install` | `FILE:pypi::ligitgays@1.0::file::pipe:[10]` |

### Case 8: pypi::bettercolors@0.1.1.graph

- **graph**: `artifacts/osp_all/pypi::bettercolors@0.1.1.graph.pt`
- **label (y/Level)**: `1`
- **triage rank (lower is more suspicious)**: base `287` -> rarity `641`

#### Top-K suspicious edges (base)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::bettercolors@0.1.1::proc::install` | `FILE:pypi::bettercolors@0.1.1::file::__pycache__/fancy_getopt.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::bettercolors@0.1.1::proc::install` | `FILE:pypi::bettercolors@0.1.1::file::__pycache__/core.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::bettercolors@0.1.1::proc::install` | `FILE:pypi::bettercolors@0.1.1::file::__pycache__/extension.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::bettercolors@0.1.1::proc::install` | `FILE:pypi::bettercolors@0.1.1::file::__pycache__/errors.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::bettercolors@0.1.1::proc::install` | `FILE:pypi::bettercolors@0.1.1::file::__pycache__/dist.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::bettercolors@0.1.1::proc::install` | `FILE:pypi::bettercolors@0.1.1::file::__pycache__/file_util.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::bettercolors@0.1.1::proc::install` | `FILE:pypi::bettercolors@0.1.1::file::__pycache__/dir_util.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::bettercolors@0.1.1::proc::install` | `FILE:pypi::bettercolors@0.1.1::file::__pycache__/dep_util.cpython-310.pyc` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::bettercolors@0.1.1::proc::install` | `FILE:pypi::bettercolors@0.1.1::file::dev/null` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::bettercolors@0.1.1::proc::install` | `FILE:pypi::bettercolors@0.1.1::file::socket:[8]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::bettercolors@0.1.1::proc::install` | `FILE:pypi::bettercolors@0.1.1::file::pipe:[8]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::bettercolors@0.1.1::proc::install` | `FILE:pypi::bettercolors@0.1.1::file::pipe:[6]` |

#### Top-K suspicious edges (rarity-adjusted)

| suspicious score | etype | src | dst |
|---:|---|---|---|
| 0.5001 | `PROC|EXEC|CMD` | `PROC:pypi::bettercolors@0.1.1::proc::install` | `CMD:cmd::bin/python3-mpipinstall--prebettercolors==0.1.1` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::bettercolors@0.1.1::proc::install` | `FILE:pypi::bettercolors@0.1.1::file::__pycache__/fancy_getopt.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::bettercolors@0.1.1::proc::install` | `FILE:pypi::bettercolors@0.1.1::file::__pycache__/core.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::bettercolors@0.1.1::proc::install` | `FILE:pypi::bettercolors@0.1.1::file::__pycache__/extension.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::bettercolors@0.1.1::proc::install` | `FILE:pypi::bettercolors@0.1.1::file::__pycache__/errors.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::bettercolors@0.1.1::proc::install` | `FILE:pypi::bettercolors@0.1.1::file::__pycache__/dist.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::bettercolors@0.1.1::proc::install` | `FILE:pypi::bettercolors@0.1.1::file::__pycache__/file_util.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::bettercolors@0.1.1::proc::install` | `FILE:pypi::bettercolors@0.1.1::file::__pycache__/dir_util.cpython-310.pyc` |
| -0.0000 | `PROC|READ|FILE` | `PROC:pypi::bettercolors@0.1.1::proc::install` | `FILE:pypi::bettercolors@0.1.1::file::__pycache__/dep_util.cpython-310.pyc` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::bettercolors@0.1.1::proc::install` | `FILE:pypi::bettercolors@0.1.1::file::dev/null` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::bettercolors@0.1.1::proc::install` | `FILE:pypi::bettercolors@0.1.1::file::socket:[8]` |
| -0.0028 | `PROC|WRITE|FILE` | `PROC:pypi::bettercolors@0.1.1::proc::install` | `FILE:pypi::bettercolors@0.1.1::file::pipe:[8]` |
