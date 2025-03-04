# DDGRL
distributed differential graph representation learning for malicious indicators identification 

![Python](https://img.shields.io/badge/Python3-3.10-brightgreen.svg) 

## Environment Setting Up:

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
    





