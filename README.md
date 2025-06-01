# DDGRL
distributed differential graph representation learning for malicious indicators identification 

![Python](https://img.shields.io/badge/Python3-3.10-brightgreen.svg) 

## Environment Setting Up:

- General Package Support

    ```
    sudo apt update
    sudo apt install -y \
    make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl git \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev libncurses-dev libgdbm-dev libnss3-dev \
    uuid-dev libedit-dev libdb-dev libexpat1-dev

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

## Graph Feature 

- For consistency, the number of node and edge should be aligned
    ```
    Max nodes per type: {'Command': 2837, 'DNS Host': 7, 'Hostnames': 7, 'IP': 79, 'Package_Name': 1, 'Path': 165142, 'Port': 6}
    Max edges per type: {'Action': 165145, 'CMD': 2837, 'DNS': 7, 'socket_host': 7, 'socket_ip': 79, 'socket_port': 6}
    ```





