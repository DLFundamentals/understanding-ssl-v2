import torch
import torch.nn as nn
import torchvision.models as models
import os
import sys
import argparse
import glob
from pathlib import Path
import pandas as pd

def load_snapshot(snapshot_path, device='cpu'):
    snapshot = torch.load(snapshot_path, map_location=device, weights_only=True)
    state_dict = snapshot['MODEL_STATE']
    epochs_trained = snapshot['EPOCHS_RUN']
    return state_dict, epochs_trained

def get_model_layers(state_dict):
    prefix = "encoder.net."
    layer_names, visited = [], set()

    for key in state_dict.keys():
        if not (key.startswith(prefix) and key.endswith(".weight")):
            continue
        
        tail = key[len(prefix):-7] # getting everything excluding prefix and .weight
        if tail.startswith("projector"):
            continue

        if tail not in visited:
            visited.add(tail)
            layer_names.append(tail)

    return sorted(layer_names)

def list_sorted_snapshots(folder: str):
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.startswith('snapshot_') and f.endswith('.pth')]
    files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return files

def frobenius_norm(w_cl, w_sup):
    num = torch.norm(w_cl - w_sup, p='fro')
    denom = 0.5 * (torch.norm(w_cl, p='fro') + torch.norm(w_sup, p='fro'))
    if denom == 0:
        return None
    
    return (num / denom).item()

def get_weights(state_dict, layer_name):
    return state_dict[f"encoder.net.{layer_name}.weight"]

def init_csv_file(output_file, sup):
    if not os.path.exists(output_file):
        os.makedirs(output_file)
    
    csv_file = os.path.join(output_file, f"dcl_vs_{sup}_weight_space.csv")
    columns = ['Epoch', 'AVG_WEIGHT_DIFF']
    if not os.path.exists(csv_file):
        df = pd.DataFrame(columns=columns)
        df.to_csv(csv_file, index=False)
    else:
        df = pd.read_csv(csv_file)
    return csv_file, df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate weight differences between contrastive and supervised learning')
    parser.add_argument('--checkpoints_dir', type=str, 
                       help='Path to checkpoints directory')
    parser.add_argument('--compare', nargs='+', default=['scl', 'nscl'],
                        choices=['scl', 'nscl', 'ce'],
                        help='Methods to compare against DCL.')
    parser.add_argument('--output_file', type=str,
                       help='Path to save CSV results')
    
    args = parser.parse_args()

    dcl_dir = os.path.join(args.checkpoints_dir, 'dcl')
    dcl_ckpts = list_sorted_snapshots(dcl_dir)

    dcl_state0, _ = load_snapshot(os.path.join(dcl_dir, dcl_ckpts[0]))
    dcl_layers = get_model_layers(dcl_state0)   

    sups = args.compare if isinstance(args.compare, list) else [args.compare]

    for sup in sups:
        sup_dir = os.path.join(args.checkpoints_dir, sup)
        csv_file, df = init_csv_file(args.output_file, sup)

        print(f"Comparing DCL vs {sup.upper()} across epochs...")
        rows = []

        for name in dcl_ckpts:
            epoch = int(name.split('_')[-1].split('.')[0])
            dcl_path = os.path.join(dcl_dir, f"snapshot_{epoch}.pth")
            sup_path = os.path.join(sup_dir, f"snapshot_{epoch}.pth")
            dcl_state, _ = load_snapshot(dcl_path)
            sup_state, _ = load_snapshot(sup_path)
            
            vals = []
            with torch.no_grad():
                for layer in dcl_layers:
                    w_cl = get_weights(dcl_state, layer)
                    w_sup = get_weights(sup_state, layer)
                    val = frobenius_norm(w_cl, w_sup)
                    if val is not None:
                        vals.append(val)

            avg_val = float(sum(vals) / len(vals)) if vals else None
            rows.append({"Epoch": epoch, "AVG_WEIGHT_DIFF": avg_val})

        if rows:
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
            df = df.sort_values(by='Epoch')
            df.to_csv(csv_file, index=False)
            print(f"Saved {len(rows)} entries to {csv_file}")
        else:
            print(f"No rows to add to {csv_file}")
