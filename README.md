# DDGRL
distributed differential graph representation learning for malicious indicators identification 

![Python](https://img.shields.io/badge/Python3-3.10-brightgreen.svg) 
![Golang](https://img.shields.io/badge/Go1.22.2-brightblue.svg) 

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
accelerate launch mgan.py
```

## Experiment Note
```
# download large download in multiple processes
sudo apt update
sudo apt install aria2
# need cookie to enable right file download
aria2c -k 1M -x 8 -s 8 \
--header="Cookie: C.20646273_auth_token=d78a9c53c67d13a66eacf0b3a3b36c7716b182907f53df1fb7a2dd2fa304e52a; username-174-93-255-152-30579=\"2|1:0|10:1749218255|29:username-174-93-255-152-30579|192:eyJ1c2VybmFtZSI6ICI1ZTIwZmZhNDY3Njk0ZGM2OWE2ZTUyZmNkMTZiZmY3YyIsICJuYW1lIjogIkFub255bW91cyBDYXJwbyIsICJkaXNwbGF5X25hbWUiOiAiQW5vbnltb3VzIENhcnBvIiwgImluaXRpYWxzIjogIkFDIiwgImNvbG9yIjogbnVsbH0=|a3c2277c844327ceb36ca27bf798682205b1c622c3da302d89b7645c4e6907ee\"; _xsrf=2|fb25189e|4a187458f44f232c65a4544cec970cb2|1749218255" \
"https://174.93.255.152:30579/files/workspace/DDGRL.zip"

```

## Graph Feature 

- For consistency, the number of node and edge should be aligned
    ```
    Max edges per type: {'Action': 128781, 'CMD': 2837, 'DNS': 7, 'socket_host': 7, 'socket_ip': 79, 'socket_port': 6}
    ```





