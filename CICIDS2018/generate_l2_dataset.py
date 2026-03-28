import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
import torch

# Your STRICT 77 Features Contract (from final_dataset.py)
CICIDS_FEATURES = [
    "protocol","flow_duration","total_fwd_packets","total_backward_packets",
    "fwd_packets_length_total","bwd_packets_length_total","fwd_packet_length_max",
    "fwd_packet_length_min","fwd_packet_length_mean","fwd_packet_length_std",
    "bwd_packet_length_max","bwd_packet_length_min","bwd_packet_length_mean",
    "bwd_packet_length_std","flow_bytes_s","flow_packets_s","flow_iat_mean",
    "flow_iat_std","flow_iat_max","flow_iat_min","fwd_iat_total","fwd_iat_mean",
    "fwd_iat_std","fwd_iat_max","fwd_iat_min","bwd_iat_total","bwd_iat_mean",
    "bwd_iat_std","bwd_iat_max","bwd_iat_min","fwd_psh_flags","bwd_psh_flags",
    "fwd_urg_flags","bwd_urg_flags","fwd_header_length","bwd_header_length",
    "fwd_packets_s","bwd_packets_s","packet_length_min","packet_length_max",
    "packet_length_mean","packet_length_std","packet_length_variance",
    "fin_flag_count","syn_flag_count","rst_flag_count","psh_flag_count",
    "ack_flag_count","urg_flag_count","cwe_flag_count","ece_flag_count",
    "down_up_ratio","avg_packet_size","avg_fwd_segment_size","avg_bwd_segment_size",
    "fwd_avg_bytes_bulk","fwd_avg_packets_bulk","fwd_avg_bulk_rate",
    "bwd_avg_bytes_bulk","bwd_avg_packets_bulk","bwd_avg_bulk_rate",
    "subflow_fwd_packets","subflow_fwd_bytes","subflow_bwd_packets",
    "subflow_bwd_bytes","init_fwd_win_bytes","init_bwd_win_bytes",
    "fwd_act_data_packets","fwd_seg_size_min","active_mean","active_std",
    "active_max","active_min","idle_mean","idle_std","idle_max","idle_min"
]

# MITRE 5-Stage Mapping for BiLSTM & HMM
# Defined by prompt: 0=Recon, 1=Initial Access, 2=PrivEsc, 3=Lateral Movement, 4=Exfiltration
# We map CICIDS18 attacks logically to these stages
STAGE_MAP = {
    'Benign': 0, 
    'FTP-BruteForce': 1, 'SSH-Bruteforce': 1, 'Brute Force -Web': 1, 
    'Brute Force -XSS': 1, 'SQL Injection': 1,
    'Infiltration': 3, # Lateral Movement roughly
    'Bot': 3,
    'DDoS attacks-LOIC-HTTP': 4, 'DDoS attacks-HOIC': 4, # Exfiltration/Impact
    'DoS attacks-GoldenEye': 4, 'DoS attacks-Hulk': 4, 
    'DoS attacks-SlowHTTPTest': 4, 'DoS attacks-Slowloris': 4
}

# Node-level categories for the Transformer 4-class categorization
# E.g., 0=Benign, 1=Bruteforce, 2=Lateral, 3=DoS/Impact
NODE_CAT_MAP = {
    'Benign': 0, 
    'FTP-BruteForce': 1, 'SSH-Bruteforce': 1, 'Brute Force -Web': 1, 
    'Brute Force -XSS': 1, 'SQL Injection': 1,
    'Infiltration': 2, 'Bot': 2,
    'DDoS attacks-LOIC-HTTP': 3, 'DDoS attacks-HOIC': 3,
    'DoS attacks-GoldenEye': 3, 'DoS attacks-Hulk': 3, 
    'DoS attacks-SlowHTTPTest': 3, 'DoS attacks-Slowloris': 3
}

def generate_l2_sequences(input_csv, out_dir="l2_dataset_tensors", chunk_size=300000, seq_len=50):
    print(f"Generating Sequence Dataset from: {input_csv}")
    os.makedirs(out_dir, exist_ok=True)
    
    # Process only a chunk to save time for huge files
    df = pd.read_csv(input_csv, skipinitialspace=True, nrows=chunk_size, low_memory=False)
    
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_')

    mapping = {
        'tot_fwd_pkts': 'total_fwd_packets', 'tot_bwd_pkts': 'total_backward_packets',
        'totlen_fwd_pkts': 'fwd_packets_length_total', 'totlen_bwd_pkts': 'bwd_packets_length_total',
        'flow_byts_s': 'flow_bytes_s', 'flow_pkts_s': 'flow_packets_s',
        'fwd_iat_tot': 'fwd_iat_total', 'bwd_iat_tot': 'bwd_iat_total',
        'fwd_header_len': 'fwd_header_length', 'bwd_header_len': 'bwd_header_length',
        'fwd_pkts_s': 'fwd_packets_s', 'bwd_pkts_s': 'bwd_packets_s',
        'pkt_len_min': 'packet_length_min', 'pkt_len_max': 'packet_length_max',
        'pkt_len_mean': 'packet_length_mean', 'pkt_len_std': 'packet_length_std',
        'pkt_len_var': 'packet_length_variance', 'fin_flag_cnt': 'fin_flag_count',
        'syn_flag_cnt': 'syn_flag_count', 'rst_flag_cnt': 'rst_flag_count',
        'psh_flag_cnt': 'psh_flag_count', 'ack_flag_cnt': 'ack_flag_count',
        'urg_flag_cnt': 'urg_flag_count', 'cwe_flag_count': 'cwe_flag_count',
        'ece_flag_cnt': 'ece_flag_count', 'pkt_size_avg': 'avg_packet_size',
        'fwd_seg_size_avg': 'avg_fwd_segment_size', 'bwd_seg_size_avg': 'avg_bwd_segment_size',
        'fwd_byts_b_avg': 'fwd_avg_bytes_bulk', 'fwd_pkts_b_avg': 'fwd_avg_packets_bulk',
        'fwd_blk_rate_avg': 'fwd_avg_bulk_rate', 'bwd_byts_b_avg': 'bwd_avg_bytes_bulk',
        'bwd_pkts_b_avg': 'bwd_avg_packets_bulk', 'bwd_blk_rate_avg': 'bwd_avg_bulk_rate',
        'subflow_fwd_pkts': 'subflow_fwd_packets', 'subflow_fwd_byts': 'subflow_fwd_bytes',
        'subflow_bwd_pkts': 'subflow_bwd_packets', 'subflow_bwd_byts': 'subflow_bwd_bytes',
        'init_fwd_win_byts': 'init_fwd_win_bytes', 'init_bwd_win_byts': 'init_bwd_win_bytes',
        'fwd_act_data_pkts': 'fwd_act_data_packets'
    }
    df = df.rename(columns=mapping)
    
    # Sort chronologically
    time_col = 'timestamp' if 'timestamp' in df.columns else 'flow_pts'
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=[time_col]).sort_values(by=time_col)

    for feature in CICIDS_FEATURES:
        if feature not in df.columns:
            df[feature] = 0.0
            
    # Group by Source IP to simulate multi-attacker independent streams
    ip_col = 'src_ip' if 'src_ip' in df.columns else 'source_ip'
    if ip_col not in df.columns:
        # If no Source IP is available in the raw chunk, we simulate one by grouping 
        # based on label to maintain sequence coherence of attacks.
        print("No Source IP found. Using Label-based pseudo-grouping for sequence generation.")
        df['simulated_src_ip'] = df['label']
        ip_col = 'simulated_src_ip'

    df_clean = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = MinMaxScaler()
    df_clean[CICIDS_FEATURES] = scaler.fit_transform(df_clean[CICIDS_FEATURES].astype(np.float32))

    df_clean['stage_label'] = df_clean['label'].map(STAGE_MAP).fillna(0).astype(int)
    df_clean['node_label'] = df_clean['label'].map(NODE_CAT_MAP).fillna(0).astype(int)

    all_windows = []
    all_node_labels = []
    all_window_stages = []
    all_severities = []
    
    grouped = df_clean.groupby(ip_col)
    
    for ip, group in grouped:
        group_len = len(group)
        if group_len < seq_len:
            # Pad with benign (0) to reach seq_len
            pad_len = seq_len - group_len
            features = group[CICIDS_FEATURES].values
            padded_features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant')
            
            node_labels = group['node_label'].values
            padded_nodes = np.pad(node_labels, (0, pad_len), mode='constant')
            
            stage = group['stage_label'].max() # Sequence classification is the max threat stage
            # Severity float [0-1] based on number of attack logs in window
            severity = np.count_nonzero(group['stage_label'].values) / seq_len
            
            all_windows.append(padded_features)
            all_node_labels.append(padded_nodes)
            all_window_stages.append(stage)
            all_severities.append(severity)
        else:
            # Chunk into non-overlapping windows of seq_len
            for i in range(0, group_len - seq_len + 1, seq_len):
                window = group.iloc[i:i+seq_len]
                all_windows.append(window[CICIDS_FEATURES].values)
                all_node_labels.append(window['node_label'].values)
                
                stage = window['stage_label'].max()
                severity = np.count_nonzero(window['stage_label'].values) / seq_len
                
                all_window_stages.append(stage)
                all_severities.append(severity)
                
    features_tensor = torch.tensor(np.array(all_windows), dtype=torch.float32)
    nodes_tensor    = torch.tensor(np.array(all_node_labels), dtype=torch.long)
    stages_tensor   = torch.tensor(np.array(all_window_stages), dtype=torch.long)
    severity_tensor = torch.tensor(np.array(all_severities), dtype=torch.float32).unsqueeze(1)
    
    print(f"Generated {len(all_windows)} windows of sequence length {seq_len}")
    print(f"Features: {features_tensor.shape}, Nodes: {nodes_tensor.shape}, Stages: {stages_tensor.shape}, Severity: {severity_tensor.shape}")

    torch.save({
        'features': features_tensor,
        'nodes': nodes_tensor,
        'stages': stages_tensor,
        'severe': severity_tensor
    }, os.path.join(out_dir, "l2_seq_dataset.pt"))
    
    with open(os.path.join(out_dir, "l2_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
        
    print("Dataset Generation Complete!")

if __name__ == "__main__":
    generate_l2_sequences("02-14-2018.csv", out_dir="../layer2_correlation/data")
