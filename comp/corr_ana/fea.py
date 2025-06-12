import pandas as pd
import ast
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import psutil
import os
from sklearn.preprocessing import OneHotEncoder


def extract_dns_features(row):
    dns_query_count = 0
    dns_host_set = set()
    dns_type_counter = {}

    for feature in ['import_DNS', 'install_DNS']:
        dns_entries = row.get(feature, [])
        if isinstance(dns_entries, (list, np.ndarray)):
            for entry in dns_entries:
                queries = entry.get("Queries", [])
                for q in queries:
                    hostname = q.get("Hostname")
                    if hostname:
                        dns_host_set.add(hostname)
                    for t in q.get("Types", []):
                        dns_type_counter[t] = dns_type_counter.get(t, 0) + 1
                dns_query_count += len(queries)

    return {
        'dns_total_queries': dns_query_count,
        'dns_unique_hosts': len(dns_host_set),
        'dns_unique_types': len(dns_type_counter)
    }

def extract_file_features(row):
    file_path_set = set()
    action_counter = {"Read": 0, "Write": 0, "Delete": 0}

    for feature in ['import_Files', 'install_Files']:
        file_entries = row.get(feature, [])
        if isinstance(file_entries, (list, np.ndarray)):
            for ent in file_entries:
                path = ent.get("Path")
                if path:
                    file_path_set.add(path)
                for act in ["Read", "Write", "Delete"]:
                    if ent.get(act):
                        action_counter[act] += 1

    return {
        'file_unique_paths': len(file_path_set),
        'file_read_count': action_counter["Read"],
        'file_write_count': action_counter["Write"],
        'file_delete_count': action_counter["Delete"]
    }

def extract_socket_features(row):
    ip_set = set()
    hostname_set = set()
    port_set = set()

    for feature in ['import_Sockets', 'install_Sockets']:
        sock_entries = row.get(feature, [])
        if isinstance(sock_entries, (list, np.ndarray)):
            for ent in sock_entries:
                ip = ent.get("Address")
                if ip and ip != "::1":
                    ip_set.add(ip)
                hostnames = ent.get("Hostnames", [])
                hostname_set.update(h for h in hostnames if h)
                port = ent.get("Port")
                if port and port != 0:
                    port_set.add(str(port))

    return {
        'socket_unique_ips': len(ip_set),
        'socket_unique_hostnames': len(hostname_set),
        'socket_unique_ports': len(port_set)
    }

def extract_command_features(row):
    cmd_total_count = 0
    cmd_arg_total = 0
    cmd_env_total = 0
    cmd_unique_set = set()

    for feature in ['import_Commands', 'install_Commands']:
        cmd_entries = row.get(feature, [])
        if isinstance(cmd_entries, (list, np.ndarray)):
            for ent in cmd_entries:
                cmd = ent.get("Command", [])
                env = ent.get("Environment", [])
                cmd_total_count += 1
                cmd_arg_total += len(cmd)
                cmd_env_total += len(env)
                if isinstance(cmd, (list, np.ndarray)) and len(cmd) > 0:
                    cmd_unique_set.add(" ".join(cmd))

    return {
        'cmd_total_count': cmd_total_count,
        'cmd_total_args': cmd_arg_total,
        'cmd_total_envs': cmd_env_total,
        'cmd_unique_commands': len(cmd_unique_set)
    }


def extract_graph_features(row):
    features = {}
    features.update(extract_dns_features(row))
    features.update(extract_file_features(row))
    features.update(extract_socket_features(row))
    features.update(extract_command_features(row))
    return pd.Series(features)


def extract_all_features(df):
    features_df = df.apply(extract_graph_features, axis=1)

    # generate one-hot encoded eco
    features_df['Ecosystem'] = df['Ecosystem'].astype(str)

    if 'Label' in df.columns:
        features_df['Label'] = df['Label']

    features_df['name_version'] = df['Name'].astype(str) + "_" + df['Version'].astype(str)

    return features_df


if __name__ == "__main__":
    data_path = Path.cwd().parent.parent.joinpath("data", "label_data.pkl")
    df = pd.read_pickle(data_path)
    all_df = extract_all_features(df)
    print(f"[Correlation Analysis] CPU memory usage (RSS): {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")
    target_columns = [
        "name_version", "Ecosystem", "dns_total_queries", 
        "dns_unique_hosts", "dns_unique_types", "file_unique_paths",
        "file_read_count", "file_write_count", "file_delete_count",
        "socket_unique_ips", "socket_unique_hostnames", "socket_unique_ports",
        "cmd_total_count", "cmd_total_args", "cmd_total_envs", "cmd_unique_commands", "Label"
    ]
    fea_df = all_df[target_columns].copy()
    print(fea_df.columns)
    fea_df.to_csv(Path.cwd().joinpath("feature_matrix.csv"), index=False)

    for col in fea_df.columns:
        print(df[col].value_counts())
    

