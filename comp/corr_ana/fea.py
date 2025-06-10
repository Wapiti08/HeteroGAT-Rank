import pandas as pd
import ast
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import psutil
import os

def safe_eval(val):
    try:
        return ast.literal_eval(val)
    except:
        return []

def extract_graph_features(row):
    def count_rw_ops(file_list):
        read, write, delete = 0, 0, 0
        for item in file_list:
            if isinstance(item, dict):
                read += int(item.get('Read', False))
                write += int(item.get('Write', False))
                delete += int(item.get('Delete', False))
        return read, write, delete

    def count_command_stats(command_list):
        total_cmds = 0
        total_args = 0
        total_envs = 0
        for cmd in command_list:
            if isinstance(cmd, dict):
                cmd_arr = cmd.get('Command', [])
                env_arr = cmd.get('Environment', [])
                total_cmds += 1
                total_args += len(cmd_arr)
                total_envs += len(env_arr)
        return total_cmds, total_args, total_envs

    features = {}

    # Parse all complex columns
    import_files = safe_eval(row['import_Files'])
    install_files = safe_eval(row['install_Files'])
    import_cmds = safe_eval(row['import_Commands'])
    install_cmds = safe_eval(row['install_Commands'])
    import_dns = safe_eval(row['import_DNS'])
    install_dns = safe_eval(row['install_DNS'])
    import_socks = safe_eval(row['import_Sockets'])
    install_socks = safe_eval(row['install_Sockets'])

    # File access
    r1, w1, d1 = ount_rw_opsc(import_files)
    r2, w2, d2 = count_rw_ops(install_files)
    features.update({
        'import_read': r1, 'import_write': w1, 'import_delete': d1,
        'install_read': r2, 'install_write': w2, 'install_delete': d2
    })

    # Commands
    c1, a1, e1 = count_command_stats(import_cmds)
    c2, a2, e2 = count_command_stats(install_cmds)
    features.update({
        'import_command_count': c1, 'import_arg_count': a1, 'import_env_count': e1,
        'install_command_count': c2, 'install_arg_count': a2, 'install_env_count': e2
    })

    # DNS & Socket
    features['import_dns_count'] = len(import_dns)
    features['install_dns_count'] = len(install_dns)
    features['import_socket_count'] = len(import_socks)
    features['install_socket_count'] = len(install_socks)

    # --- Ecosystem Encoding ---
    eco_str = str(row.get('Ecosystem', 'unknown'))
    eco_code = hash(eco_str) % 1000  # hash-based encoding (or map manually if needed)
    features['eco_code'] = eco_code

    return pd.Series(features)

if __name__ == "__main__":
    data_path = Path.cwd().parent.parent.joinpath("data", "label_data.pkl")
    df = pd.read_pickle(data_path)
    for _, row in df.iterrows():
        features =  extract_graph_features(row)
        for key, value in features.items():
            df.at[row.name, key] = value

    print(f"[Correlation Analysis] CPU memory usage (RSS): {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")


    for col in df.columns:
        print(df[col].value_counts())
    

