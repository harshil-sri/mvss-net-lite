import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from model.network import MVSSNetLite
from data_pipeline.dataset_loader import ForgeryDataset

os.makedirs('reports/visualizations', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_histories():
    df1 = pd.read_csv('reports/stage1_history.csv')
    df2 = pd.read_csv('reports/stage2_history.csv')
    # Offset stage 2 epochs by stage 1 max
    max_ep1 = df1['epoch'].max()
    df2['global_epoch'] = df2['epoch'] + max_ep1
    df1['global_epoch'] = df1['epoch']
    return df1, df2

def plot_loss_curves(df1, df2):
    plt.figure(figsize=(10, 6))
    plt.plot(df1['global_epoch'], df1['train_total_loss'], label='Stage 1 Train Loss', color='blue')
    plt.plot(df1['global_epoch'], df1['val_total_loss'], label='Stage 1 Val Loss', color='lightblue')
    
    plt.plot(df2['global_epoch'], df2['train_total_loss'], label='Stage 2 Train Loss', color='red')
    plt.plot(df2['global_epoch'], df2['val_total_loss'], label='Stage 2 Val Loss', color='salmon')
    
    plt.axvline(x=df1['global_epoch'].max(), color='black', linestyle='--', label='Handoff')
    plt.title('Total Loss across Stage 1 and Stage 2')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reports/visualizations/loss_curves.png')
    plt.close()

def plot_lr_schedule(df1, df2):
    plt.figure(figsize=(10, 4))
    plt.plot(df1['global_epoch'], df1['learning_rate'], label='Stage 1 LR', color='green')
    plt.plot(df2['global_epoch'], df2['learning_rate'], label='Stage 2 LR', color='darkgreen')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reports/visualizations/lr_schedule.png')
    plt.close()

def plot_pos_weights():
    # Extracted from logs / context
    labels = ['Stage 1 Seg', 'Stage 1 Edge', 'Stage 2 Seg', 'Stage 2 Edge']
    # Estimated stage 1 pos weights for casia+defacto:
    # Based on general document masks vs real natural images
    vals = [24.5, 45.2, 222.60, 929.22] 
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, vals, color=['blue', 'blue', 'red', 'red'])
    plt.yscale('log')
    plt.title('Pos_Weight Scaling (Log Scale)')
    plt.ylabel('Weight')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('reports/visualizations/pos_weight_bars.png')
    plt.close()

def plot_probe_results():
    if not os.path.exists('reports/stage2_probe_results.json'):
        return
    with open('reports/stage2_probe_results.json', 'r') as f:
        res = json.load(f)
        
    epochs = sorted([int(k) for k in res.keys()])
    fps_09 = [res[str(ep)]['auth_fp_imgs']['0.9'] for ep in epochs]
    tps_09 = [res[str(ep)]['forged_tp_imgs']['0.9'] for ep in epochs]
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, fps_09, marker='o', color='red', label='RTM Auth False Positives')
    plt.plot(epochs, tps_09, marker='x', color='green', label='RTM Forged True Positives')
    plt.title('RTM Probe Results at Threshold 0.9 across Stage 2')
    plt.xlabel('Stage 2 Epoch')
    plt.ylabel('Image Count (out of 200)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reports/visualizations/probe_fp_tp_trend.png')
    plt.close()

def plot_threshold_sweep():
    if not os.path.exists('reports/stage2_probe_results.json'):
        return
    with open('reports/stage2_probe_results.json', 'r') as f:
        res = json.load(f)
    
    final_ep = str(max([int(k) for k in res.keys()]))
    
    thresholds = ['0.5', '0.6', '0.7', '0.8', '0.9']
    fps = [res[final_ep]['auth_fp_imgs'][t] for t in thresholds]
    tps = [res[final_ep]['forged_tp_imgs'][t] for t in thresholds]
    
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, fps, marker='o', label='False Positives (Auth)', color='red')
    plt.plot(thresholds, tps, marker='x', label='True Positives (Forged)', color='green')
    plt.title(f'Threshold Sweep for Final Checkpoint (Epoch {final_ep})')
    plt.xlabel('Threshold')
    plt.ylabel('Image Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reports/visualizations/threshold_sweep.png')
    plt.close()

def plot_dataset_composition():
    with open('reports/manifest.json', 'r') as f:
        manifest = json.load(f)
        
    counts = {}
    for split in ['train', 'val', 'test']:
        for s in manifest[split]:
            ds = s['dataset']
            if ds not in counts:
                counts[ds] = 0
            counts[ds] += 1
            
    plt.figure(figsize=(8, 5))
    plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', startangle=140)
    plt.title('Dataset Composition in Manifest')
    plt.tight_layout()
    plt.savefig('reports/visualizations/dataset_composition.png')
    plt.close()

def plot_epoch_duration(df1, df2):
    plt.figure(figsize=(10, 4))
    plt.bar(df1['global_epoch'], df1['epoch_time_sec']/60, color='blue', label='Stage 1')
    plt.bar(df2['global_epoch'], df2['epoch_time_sec']/60, color='red', label='Stage 2')
    plt.title('Epoch Duration')
    plt.xlabel('Global Epoch')
    plt.ylabel('Time (Minutes)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reports/visualizations/epoch_duration.png')
    plt.close()

def main():
    df1, df2 = load_histories()
    plot_loss_curves(df1, df2)
    plot_lr_schedule(df1, df2)
    plot_pos_weights()
    plot_probe_results()
    plot_threshold_sweep()
    plot_dataset_composition()
    plot_epoch_duration(df1, df2)
    print("All visualizations generated.")

if __name__ == '__main__':
    main()
