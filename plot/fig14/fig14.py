import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from matplotlib import rcParams
from matplotlib import transforms

# rcParams['font.family'] = 'Times New Roman'


import csv
import json


def parse_csv(file_path):

    data = {}
    
  
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
       
            batch_size = row['Batch Size']
            seq_length = row['Sequence Length']
            key = f"({batch_size},{seq_length})"
            
        
            if key not in data:
                data[key] = {}
            
      
            model = row['Model']
            
           
            if model not in data[key]:
                data[key][model] = {}
            
           
            for column in ['Analytical Model', 'Hash Encoding', 'Numercial Decoding', 'Reward Algorithm', 'STOF']:
                
                value = float(row[column]) if '.' in row[column] else int(row[column])
                data[key][model][column] = value
    
    return data


import argparse
parser = argparse.ArgumentParser(description='Plot overhead analysis')
parser.add_argument('--file_path1', default="../../data/Overhead_Analysis/overhead_analysis.csv",)
args = parser.parse_args()
data =parse_csv(args.file_path1)


newdata = {}
for bs_seq in data:
    newdata[bs_seq] = {}
    for model in data[bs_seq]:
        total = data[bs_seq][model]["STOF"]
        newdata[bs_seq][model] = {
            "Analytical Model": data[bs_seq][model]["Analytical Model"] / total,
            "Hash Encoding": data[bs_seq][model]["Hash Encoding"] / total,
            "Numercial Decoding": data[bs_seq][model]["Numercial Decoding"] / total,
            "Reward Algorithm": data[bs_seq][model]["Reward Algorithm"] / total
        }


categories = ['Reward Algorithm', 'Numercial Decoding', 'Hash Encoding', 'Analytical Model']
colors = ['#afc8ea', '#dff1d7', '#f2edb7', '#f4b1c9']
model_order = ['Bert-small', 'Bert-base', 'Bert-large', 'GPT', 'T5']
bs_seq_order = ['(1, 128)', '(8, 512)', '(16, 2048)']


fig, ax = plt.subplots(figsize=(25, 7))

bar_width = 0.16
model_gap = 0.150
group_gap = 0.3

positions = []
x_labels = []
group_boundaries = []

current_pos = 0
for idx, bs_seq in enumerate(bs_seq_order):
    group_positions = []
    for i in range(len(model_order)):
        group_positions.append(current_pos + i * (bar_width + model_gap))

    positions.extend(group_positions)
    group_width = (len(model_order) - 1) * (bar_width + model_gap) + bar_width
    group_boundaries.append(current_pos + group_width / 2)
    current_pos += group_width + group_gap
    x_labels.extend(model_order)


bottom = np.zeros(len(positions))
for i, cat in enumerate(categories):
    heights = []
    for bs_seq in bs_seq_order:
        for model in model_order:
            heights.append(newdata[bs_seq][model][cat] * 100)
    ax.bar(positions, heights, bar_width,
           bottom=bottom, color=colors[i],
           edgecolor='black', linewidth=0.5,
           label=cat,zorder=3)
    bottom += heights


ax.set_ylabel('Percentage', fontsize=50)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.set_ylim(0, 3)
ax.tick_params(axis='y', labelsize=38)


ax.yaxis.grid(True, linestyle=(0, (10, 5)), linewidth=1)


ax.set_xticks([]) 
ax.set_xticklabels([]) 


x_labelss= ['BERT-Small', 'BERT-Base', 'BERT-Large', 'GPT', 'T5']
for pos, label,i in zip(positions, x_labels,range(15)):
    ax.text(
        x=pos+0.06,                         
        y=-0.016,                          
        s=x_labelss[i%5],  
        rotation=45,   
        ha='right',
        va='top',       
        fontsize=33,  
        transform=ax.get_xaxis_transform() 
    )

for label in ax.get_xticklabels():
    x_pos = label.get_position()[0]
    offset = transforms.ScaledTranslation(-0.67, 0, ax.transData)
    label.set_transform(ax.transData + offset)


for center, label in zip(group_boundaries, bs_seq_order):
    fig.text(
        x=center-0.07,             
        y=-0.44,   
        s=label, 
        ha='center',
        va='top',  
        fontsize=53,  
        transform=ax.get_xaxis_transform(), 
        clip_on=False
    )


for boundary in group_boundaries[:-1]:
    ax.axvline(
        boundary + group_gap / 2+0.62,
        color='grey',
        linestyle='--',
        linewidth=5,
        ymax=0.98
    )


handles, labels = ax.get_legend_handles_labels()
handles = handles[::-1] 
labels=labels[::-1]
legend = fig.legend(
    handles=handles,
    labels=labels,
    ncol=2,
    bbox_to_anchor=(0.5, 1.34),
    loc='upper center',
    fontsize=55,
    edgecolor='black',
    framealpha=1,
    frameon=False
)
ax.margins(x=0.03)
plt.tight_layout()
plt.savefig('fig14.pdf', dpi=450, bbox_inches='tight')