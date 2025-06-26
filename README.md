# DDGRL
distributed differential graph representation learning for malicious indicators identification 

![Python](https://img.shields.io/badge/Python3-3.10-brightgreen.svg) 
![Golang](https://img.shields.io/badge/Go1.22.2-brightblue.svg) 
![CUDA](https://img.shields.io/badge/CUDA12.4-brightred.svg) 

## Environment Setting Up:

- General Package Support

    ```
    sudo apt update
    
    sudo apt install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev

    ```

- Download Pyenv:
    ```
    # for linux
    curl https://pyenv.run | bash

    # add the following lines to ~/.profile and ~/.bashrc
    export PYENV_ROOT="$HOME/.pyenv"
    [[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init - bash)"
    
    # add the following to ~/.bashrc:
    eval "$(pyenv virtualenv-init -)"
    
    # reactivate bash files
    source ~/.bashrc
    source ~/.profile
    ```

- Configure Python Version
    ```
    pyenv install 3.10.1
    pyenv global 3.10.1
    # activate environment
    pyenv virtualenv 3.10.1 DDGRL
    pyenv local DDGRL
    # install libraries
    pip3 install -r requirements.txt
    # for quick test
    pip3 install pandas==2.2.3 tqdm==4.67.1 sentence-transformers==3.4.1 matplotlib==3.10.3 seaborn==0.13.2 shap==0.47.2 xgboost==3.0.2 accelerate==1.7.0 ace_tools==0.0 torch==2.6.0 torch-geometric==2.6.1 dask==2023.5.0 dask[distributed]==2023.5.0

    ```

- Other packages
    ```
    sudo apt-get install libffi-dev # avoid _ctypes error
    sudo apt-get install libbz2-dev # avoid _bz2 error
    sudo apt install golang-go

    ```

## Distributed Configuration

```
accelerate config
# this machine
# multi-GPU
# 1 machine
# default choice for other options
# 8 GPUs
# default choice for other options
# !no mixed precision --- in order to run sparse matrix

```

## Graph Representation

- Individual Package Graph Representation (Undirected Graph):

    - pack information:
        
        central_node (root node): {name}_{version}

        central_node_attr: {ecosystem}

    - import_Files (install_Files):

        -- edge --> leaf node: ---{True Action: Read and Write}---> Path

    - import_Sockets (install_Sockets):

        remove port with 0 and address with '::1' and blank hostname / address / port

        {IP} - {Hostname} - {Port}
    
    - import_Commands (install_Commands):

        {command}: concat all strings
        
    
    - import_DNS (install_DNS):

        leaf node: {Hostname}

        left node attr: {Types} in list

## How to use

- data generation:
```
python3 data_create.py
```

- graph analysis:
```
python3 mgan.py
```

- entropy analysis:
```
# under the golang project workdir
go mod init go_entropy_ana
go mod tidy
go run *.go

```

- Single GPU Training:
```
CUDA_VISIBLE_DEVICES=2 python3 mgan.py

```

- Multiple GPUs Training:
```
accelerate launch dgan.py
# optional: add port conflict from dask and accelerate -> automatically use next available port
nohup accelerate launch --main_process_port=0 dgan.py >output.txt 2>&1 &
```

## Experiment Note
```
# download large download in multiple processes
sudo apt update
sudo apt install aria2
# need cookie to enable right file download
aria2c -k 1M -x 8 -s 8 \
--header="Cookie: username-mixing-graphics-agenda-librarian-trycloudflare-com=\"2|1:0|10:1750668312|59:username-mixing-graphics-agenda-librarian-trycloudflare-com|200:eyJ1c2VybmFtZSI6ICJlYmQ5YjgwZDk0YTM0ZjIwYWI0NDM0NjE5MTlhODU3YiIsICJuYW1lIjogIkFub255bW91cyBBbWFsdGhlYSIsICJkaXNwbGF5X25hbWUiOiAiQW5vbnltb3VzIEFtYWx0aGVhIiwgImluaXRpYWxzIjogIkFBIiwgImNvbG9yIjogbnVsbH0=|4a393cbf532b62f3e2d481a6f394a6cec07587f1e95ffc80b8cafe11cb304cb3\"; _xsrf=2|854af5bb|5b2a4314559e61e83185255caa3ccea4|1750668312" \
"https://mixing-graphics-agenda-librarian.trycloudflare.com/files/workspace/DDGRL.zip?_xsrf=2%7C854af5bb%7C5b2a4314559e61e83185255caa3ccea4%7C1750668312"

# download from google drive
pip3 install gdown
gdown --folder "{shared_link}"

```

## Graph Feature 

- For consistency, the number of node and edge should be aligned
    ```
    Max edges per type: {'Action': 128781, 'CMD': 2837, 'DNS': 7, 'socket_host': 7, 'socket_ip': 79, 'socket_port': 6}
    ```

## Next Work
- build ID to projection with: (node_type, value) 
    ```
    self.reverse_node_id_map = {
        global_id: f"{node_type}:{value}"
        for node_type, id_map in local_to_global.items()
        for global_id, value in zip(id_map.values(), id_map.keys())
    }
    ```



