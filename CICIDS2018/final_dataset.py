import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler

# 1. Your STRICT 77 Features Contract
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

# 2. MITRE 5-Stage Mapping for the BiLSTM
MITRE_MAP = {
    'Benign': 0, 
    'FTP-BruteForce': 1, 'SSH-Bruteforce': 1, 'Brute Force -Web': 1, 
    'Brute Force -XSS': 1, 'SQL Injection': 1,
    'Infiltration': 2, 
    'Bot': 3, 
    'DDoS attacks-LOIC-HTTP': 4, 'DDoS attacks-HOIC': 4, 
    'DoS attacks-GoldenEye': 4, 'DoS attacks-Hulk': 4, 
    'DoS attacks-SlowHTTPTest': 4, 'DoS attacks-Slowloris': 4
}

def clean_and_enforce_schema(input_csv: str, output_csv: str):
    print(f"Loading raw dataset: {input_csv}...")
    df = pd.read_csv(input_csv, skipinitialspace=True)
    
    # Standardize raw names for mapping (lowercase, remove special chars)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_')

    # Mapping raw 2018 abbreviations to your strict 77-feature schema
    mapping = {
        'tot_fwd_pkts': 'total_fwd_packets',
        'tot_bwd_pkts': 'total_backward_packets',
        'totlen_fwd_pkts': 'fwd_packets_length_total',
        'totlen_bwd_pkts': 'bwd_packets_length_total',
        'flow_byts_s': 'flow_bytes_s',
        'flow_pkts_s': 'flow_packets_s',
        'fwd_iat_tot': 'fwd_iat_total',
        'bwd_iat_tot': 'bwd_iat_total',
        'fwd_header_len': 'fwd_header_length',
        'bwd_header_len': 'bwd_header_length',
        'fwd_pkts_s': 'fwd_packets_s',
        'bwd_pkts_s': 'bwd_packets_s',
        'pkt_len_min': 'packet_length_min',
        'pkt_len_max': 'packet_length_max',
        'pkt_len_mean': 'packet_length_mean',
        'pkt_len_std': 'packet_length_std',
        'pkt_len_var': 'packet_length_variance',
        'fin_flag_cnt': 'fin_flag_count',
        'syn_flag_cnt': 'syn_flag_count',
        'rst_flag_cnt': 'rst_flag_count',
        'psh_flag_cnt': 'psh_flag_count',
        'ack_flag_cnt': 'ack_flag_count',
        'urg_flag_cnt': 'urg_flag_count',
        'cwe_flag_count': 'cwe_flag_count',
        'ece_flag_cnt': 'ece_flag_count',
        'pkt_size_avg': 'avg_packet_size',
        'fwd_seg_size_avg': 'avg_fwd_segment_size',
        'bwd_seg_size_avg': 'avg_bwd_segment_size',
        'fwd_byts_b_avg': 'fwd_avg_bytes_bulk',
        'fwd_pkts_b_avg': 'fwd_avg_packets_bulk',
        'fwd_blk_rate_avg': 'fwd_avg_bulk_rate',
        'bwd_byts_b_avg': 'bwd_avg_bytes_bulk',
        'bwd_pkts_b_avg': 'bwd_avg_packets_bulk',
        'bwd_blk_rate_avg': 'bwd_avg_bulk_rate',
        'subflow_fwd_pkts': 'subflow_fwd_packets',
        'subflow_fwd_byts': 'subflow_fwd_bytes',
        'subflow_bwd_pkts': 'subflow_bwd_packets',
        'subflow_bwd_byts': 'subflow_bwd_bytes',
        'init_fwd_win_byts': 'init_fwd_win_bytes',
        'init_bwd_win_byts': 'init_bwd_win_bytes',
        'fwd_act_data_pkts': 'fwd_act_data_packets'
    }
    df = df.rename(columns=mapping)

    # 3. Sort chronologically to preserve Layer 2 sequence validity
    time_col = 'timestamp' if 'timestamp' in df.columns else 'flow_pts'
    if time_col in df.columns:
        print("Sorting chronologically...")
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=[time_col]).sort_values(by=time_col)

    # 4. ENFORCE SCHEMA: Filter exclusively to the 77 features + Label
    # Any column not in your CICIDS_FEATURES list is dropped immediately.
    final_columns = []
    for feature in CICIDS_FEATURES:
        if feature in df.columns:
            final_columns.append(feature)
        else:
            print(f"WARNING: Feature '{feature}' missing from raw CSV. Filling with zeros.")
            df[feature] = 0.0
            final_columns.append(feature)
            
    # Reorder DataFrame to strictly match the 77-feature array order
    df_features = df[CICIDS_FEATURES]

    print("Cleaning Infinity and NaN values...")
    df_features = df_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Map Labels
    print("Mapping MITRE Attack Categories...")
    attack_cat = df['label'].map(MITRE_MAP).fillna(0).astype(int)
    
    # Combine back into final dataframe
    df_final = pd.concat([df_features, attack_cat.rename('attack_cat')], axis=1)

    print("Removing duplicate sequences...")
    df_final = df_final.drop_duplicates()

    print("Fitting Scaler for Layer 1 Production...")
    scaler = MinMaxScaler()
    df_final[CICIDS_FEATURES] = scaler.fit_transform(df_final[CICIDS_FEATURES].astype(np.float32))
    
    os.makedirs("models", exist_ok=True)
    with open("models/layer1_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"Saving pristine dataset to {output_csv}...")
    df_final.to_csv(output_csv, index=False)
    
    # Final Validation
    assert list(df_final.columns[:-1]) == CICIDS_FEATURES, "COLUMN ORDER MISMATCH!"
    print(f"Success! Dataset shape: {df_final.shape}")
    print("Columns exactly match the 77 required CICIDS_FEATURES.")

if __name__ == "__main__":
    clean_and_enforce_schema("raw_cicids2018.csv", "master_dataset/cicids_cleaned_strict.csv")