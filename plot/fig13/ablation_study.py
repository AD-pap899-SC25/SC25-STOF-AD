import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import re
import matplotlib as mpl


def parse_file(file_path, is_mcfuser=False):
    data = {}
    current_model = None

    with open(file_path, 'r') as f:
        for line in f:

            model_match = re.search(r'\[model\] Unified bench test for (\w+)', line)
            if model_match:
                current_model = model_match.group(1)
                continue

            if is_mcfuser:
                submodel_match = re.search(r'\[(\w+)\] Unified bench test for', line)
                if submodel_match:
                    current_model = submodel_match.group(1)
                    continue


            pattern = r'.*?bs:(\d+).*?seq:(\d+).*?\|\s+([\w\s]+)\s*:\s*([\d.]+)'
            match = re.search(pattern, line)
            if match:
                bs = int(match.group(1))
                seq = int(match.group(2))
                method = match.group(3).strip()
                time = float(match.group(4))

                if is_mcfuser:
                    method = "MCFuser"

                if bs not in data:
                    data[bs] = {}
                if seq not in data[bs]:
                    data[bs][seq] = {}
                if current_model not in data[bs][seq]:
                    data[bs][seq][current_model] = {}

                data[bs][seq][current_model][method] = time
    return data
def parse_file2(file_path, is_mcfuser=False):
    data = {}
    current_model = None

    with open(file_path, 'r') as f:
        for line in f:

            model_match = re.search(r'e2e (\w+)', line)
            if model_match:
                current_model = model_match.group(1)

            if is_mcfuser:
                submodel_match = re.search(r'$$\w+$$', line)
                if submodel_match:
                    continue

            pattern = r'.*?bs:(\d+).*?seq:(\d+).*?\|\s+([\w\s]+)\s*:\s*([\d.]+)'
            match = re.search(pattern, line)
            if match and current_model:
                bs = int(match.group(1))
                seq = int(match.group(2))
                method = match.group(3).strip()
                time = float(match.group(4))

                if is_mcfuser:
                    method = "MCFuser"

                data.setdefault(bs, {}).setdefault(seq, {}).setdefault(current_model, {})[method] = time

    return data

import argparse
parser = argparse.ArgumentParser(description='Plot ablation_study')
parser.add_argument('--file_path1',default="../../data/Ablation_Study/ablation-base.txt", required=False)
args = parser.parse_args()
data1 = parse_file2(args.file_path1)


def combinedData(data1):
    combined_data = {}
    target_pairs = [
        (1, 128),
        (8, 512),
        (16, 2048),
    ]


    methods_mapping = {
        'Torch Native': ('data1', 'Torch Native'),
        'STOF MHA': ('data1', 'STOF MHA'),
        'STOF Compiled': ('data1', 'STOF Compiled'),
        'STOF': ('data1', 'STOF'),


    }
    for bs in [1, 8, 16]:
        combined_data[bs] = {}
        target_seqs = [seq for b, seq in target_pairs if b == bs]

        for seq in target_seqs:
            combined_data[bs][seq] = {}

            for model in ['bert_small', 'bert_base', 'bert_large', 'gpt', 't5']:
                combined_data[bs][seq][model] = {}

                for method, (source, key) in methods_mapping.items():
                    if source == 'data1':
                        src_data = data1
                        
                    val = src_data.get(bs, {}).get(seq, {}).get(model, {}).get(key, 0)
                    combined_data[bs][seq][model][method] = val
    return combined_data

combined_data1=combinedData(data1)



def normalizedData(combined_data):
    normalized_data = {}
    for bs in combined_data:
        normalized_data[bs] = {}
        for seq in combined_data[bs]:
            normalized_data[bs][seq] = {}
            for model in combined_data[bs][seq]:
                model_data = combined_data[bs][seq][model]
                tn_time = model_data.get('Torch Native', 0)

                normalized = {}
                for method, value in model_data.items():
                    if tn_time > 0 and value > 0:
                        normalized[method] = tn_time / value
                    else:
                        normalized[method] = 0
                        
                if tn_time > 0:
                    normalized['Torch Native'] = 1.0

                normalized_data[bs][seq][model] = normalized
    return normalized_data

normalized_data1=normalizedData(combined_data1)


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 25

MODELS = ['bert_small', 'bert_base', 'bert_large', 'gpt', 't5']
models= ['BERT-Small', 'BERT-Base', 'BERT-Large', 'GPT', 'T5']
BS_SEQ_PAIRS = [(1, 128), (8, 512), (16, 2048)]
METHODS = ['Torch Native', 'STOF MHA', 'STOF Compiled', 'STOF']

COLORS = ['#999696', '#afc8ea','#dff1d7',  '#d06c5a']
LABEL_NAMES = ['PyTorch Native', 'Only Unified MHA', 'Only Operator Fusion', 'Unified MHA+Operator Fusion']



def extract_data(normalized_data):
    return {
        (bs, seq): {model: [normalized_data[bs][seq][model][method] for method in METHODS]
                    for model in MODELS}
        for bs, seq in BS_SEQ_PAIRS
    }


data1 = extract_data(normalized_data1)



def plot_subplot(ax, data):
    n_models = len(MODELS)
    n_methods = len(METHODS)
    bar_width =2
    model_spacing = 2.0 
    
    group_width = n_models * (n_methods * bar_width) + (n_models - 1) * model_spacing
    group_spacing = 5 #
    
    for pair_idx, (bs, seq) in enumerate(BS_SEQ_PAIRS):
        x_group_start = pair_idx * (group_width + group_spacing)

        for model_idx, model in enumerate(MODELS):
            
            x_model_start = x_group_start + model_idx * (n_methods * bar_width + model_spacing)

            values = data[(bs, seq)][model]

            
            for method_idx, value in enumerate(values):
                x = x_model_start + method_idx * bar_width

                rect = ax.bar(x, value, width=bar_width,
                              color=COLORS[method_idx],
                              edgecolor='black',
                              linewidth=0.8,zorder=3)

                if value > 4:
                    ax.text(x - 1.1, 3.9, 
                            f"{value:.1f}",
                            ha='right', va='top',
                            fontsize=25,
                            fontname='Times New Roman',
                            color='black')


    for i in range(1, len(BS_SEQ_PAIRS)):
        if i==1:
            offset_xian=-0.95
        else:
            offset_xian = -0.51
        x = i * (group_width + group_spacing+offset_xian) - group_spacing / 2
        ax.axvline(x, color='gray', linestyle='--', linewidth=3)

    
    ax.set_ylabel('Speedup', fontsize=30)
    ax.set_ylim(0, 4)
    ax.set_yticks([0,1,2,3,4])
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.grid(True, linestyle=(0, (10, 5)), linewidth=1)
    ax.set_xticks([]) 
    ax.margins(x=0.01)

    total_width = len(BS_SEQ_PAIRS) * (group_width + 2) - 2
    label_y_positions = {
        'model': -0.05,
        'bs_seq': -1.8  ,
    }
    
    for pair_idx, (bs, seq) in enumerate(BS_SEQ_PAIRS):
        x_center = pair_idx * (group_width + 2) + group_width / 2

        bsseqoffent = 0
        if (bs, seq) == (1,128):
            bsseqoffent=1.89
        elif (bs, seq) == (8, 512):
            bsseqoffent = -1.1
        elif (bs, seq) == (16,2048):
            bsseqoffent = -5.3
        ax.text(x_center-bsseqoffent, label_y_positions['bs_seq'],

                f'({bs}, {seq})',
                ha='center', va='top',fontsize = 35)

        model_x_centers = [x_center - group_width / 2 + (i + 0.5) * (n_methods + 0.5)
                           for i in range(n_models)]
        for x, model in zip(model_x_centers, models):
            offset = 0
            if(model=="BERT-Small") and (bs, seq) == (1,128):
                offset=4.8
            elif(model=="BERT-Base") and (bs, seq) == (1,128):
                offset = -1.4
            elif(model=="BERT-Large") and (bs, seq) == (1,128):
                offset = -6.2
            elif (model == "GPT") and (bs, seq) == (1, 128):
                offset = -16.3
            elif (model == "T5") and (bs, seq) == (1, 128):
                offset = -23
            elif (model == "BERT-Small") and (bs, seq) == (8,512):
                offset = 1.8
            elif(model=="BERT-Base") and (bs, seq) == (8,512):
                offset =-4.2
            elif(model=="BERT-Large") and (bs, seq) == (8,512):
                offset = -9.3
            elif (model == "GPT") and (bs, seq) == (8,512):
                offset = -19.3
            elif (model == "T5") and (bs, seq) == (8, 512):
                offset =-26.1
            elif (model == "BERT-Small") and (bs, seq) == (16,2048):
                offset = -1
            elif (model == "BERT-Base") and (bs, seq) == (16,2048):
                offset =-7.4
            elif (model == "BERT-Large") and (bs, seq) == (16,2048):
                offset = -12.2
            elif (model == "GPT") and (bs, seq) == (16,2048):
                offset = -22.2
            elif (model == "T5") and (bs, seq) == (16,2048):
                offset = -29
            ax.text(x-offset, label_y_positions['model'],
                    model.replace('_', ' '),
                    ha='center', va='top', fontsize = 22,rotation=45)



fig, (ax1) = plt.subplots(1, 1, figsize=(20, 3.9), dpi=300, gridspec_kw={'wspace': 0.08}  )

plot_subplot(ax1, data1)


legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor='black')
                   for c in COLORS]
fig.legend(legend_elements, LABEL_NAMES,
           loc='upper center',
           ncol=2,
           frameon=False,
           fontsize=33,

           bbox_to_anchor=(0.5, 1.31))

plt.subplots_adjust(wspace=0.12)
plt.savefig('5-ablation.pdf',
           bbox_inches='tight',
           dpi=300,
           facecolor='white')

plt.close()
