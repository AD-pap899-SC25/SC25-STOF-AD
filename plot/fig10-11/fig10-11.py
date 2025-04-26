import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
import numpy as np
import re
import argparse
import os


# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 30

def parse_file(file_path, onlymc_file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    with open(onlymc_file_path, 'r') as f:
        onlymc_lines = f.readlines()
    combined_lines = lines + onlymc_lines
    
    data = {}
    bs_order = [1, 8, 16] 
    seq_order = [128, 256, 512, 1024, 2048, 4096]
    
    for line in combined_lines:
        if re.match(r'\s*bs:\d+', line):
            match = re.match(r'\s*bs:(\d+)\s*\|\s*h_num:\d+\s*\|\s*seq:(\d+)\s*\|\s*(.+?)\s*:\s*([\d.]+)\s*ms', line)
            if match:
                bs = int(match.group(1))
                seq = int(match.group(2))
                method = match.group(3).strip()
                time = float(match.group(4))
                
                if bs not in data:
                    data[bs] = {}
                if seq not in data[bs]:
                    data[bs][seq] = {'FlashAttn2': None, 'Torch Naive': None, 
                                    'FlexAttn': None, 'MCFuser': None, 
                                    'Our Kernel': None, 'ByteTrans': None}
                data[bs][seq][method] = time
    

    for bs in bs_order:
        if bs not in data:
            data[bs] = {}
        for seq in seq_order:
            if seq not in data[bs]:
                data[bs][seq] = {k: None for k in data[bs].get(seq_order[0], {})}
    return data

def normalize_data(data):
    normalized = {}
    bs_order = [1, 8, 16]
    seq_order = [128, 256, 512, 1024, 2048, 4096]
    
    for bs in bs_order:
        normalized[bs] = {}
        for seq in seq_order:
            methods = data.get(bs, {}).get(seq, {})
            valid_times = [t for t in methods.values() if t is not None]
            
          
            base_time = methods.get('Torch Naive', 0.1)  
            
            normalized_entry = {}
            for method, time in methods.items():
                if time is not None and base_time != 0:
                    normalized_entry[method] = round(base_time / time, 1)
                else:
                    normalized_entry[method] = 0  
            normalized[bs][seq] = normalized_entry
    return normalized

def plot_performance(data_list, subtitles, gpu_info):
    fig, axs = plt.subplots(2, 2, figsize=(40, 10))
    plt.subplots_adjust(left=0.03, right=0.98, wspace=0.05,hspace=1.1)
    axs = axs.flatten()
    
    bar_width = 0.5
    bs_order = [1, 8, 16]
    seq_order = [128, 256, 512, 1024, 2048, 4096]

    colors = {
        'Torch Naive': '#999696',
        'MCFuser': '#f4b1c9',
        'ByteTrans': '#f2edb7',
        'FlashAttn2': '#dff1d7',
        'FlexAttn': '#afc8ea',
        'Our Kernel': '#d06c5a'
    }
    seq_spacing = 3.5 
    
    def add_labels_with_cutoff(rects, ax, cutoff):
        for rect in rects:
            actual_height = rect.get_height()
            if actual_height > cutoff - 0.01:
               
                ax.text(rect.get_x() - bar_width / 6,
                        cutoff * 0.98,
                        f'{actual_height:.1f}',
                        ha='right',
                        va='top',
                        fontsize=24,
                        # fontname='Times New Roman',
                        color='black')
    
    for plot_idx, (data, title) in enumerate(zip(data_list, subtitles)):
        ax = axs[plot_idx]
        x_ticks = []
        x_labels = []
        all_x = []  
        seq_labels = []  

       
        batch_gap = 1  # 根据需要调整该值
        step = len(seq_order) * seq_spacing + batch_gap
    
        x_base = np.arange(0, len(bs_order) * step, step)
    
        for bs_idx, bs in enumerate(bs_order):
           
            middle_x = x_base[bs_idx] + (len(seq_order)-1)*seq_spacing/2
            x_ticks.append(middle_x)
            x_labels.append(f'batch size={bs}')
            
            
            for seq_idx, seq in enumerate(seq_order):
                x_pos = x_base[bs_idx] + seq_idx*seq_spacing
                all_x.append(x_pos)
                seq_labels.append(str(seq))

        for i in range(1, len(bs_order)):
            prev_base = x_ticks[i-1]
            current_base = x_ticks[i]
            line_x = (prev_base + current_base)/2
            ax.axvline(x=line_x, color='gray', linestyle='--', linewidth=3)


        for method_idx, method in enumerate(['Torch Naive', 'MCFuser', 'ByteTrans', 
                                            'FlashAttn2', 'FlexAttn', 'Our Kernel']):
            x_offset = (method_idx - 2.5) * bar_width 
            for global_idx, (bs, seq) in enumerate([(b,s) for b in bs_order for s in seq_order]):
                bs_idx = bs_order.index(bs)
                seq_idx = seq_order.index(seq)
                x_pos = x_base[bs_idx] + seq_idx*seq_spacing + x_offset
                
                value = data[bs][seq].get(method, 0)
                color = colors[method]

               
                if value > 0:
                    rects = ax.bar(x_pos, value, bar_width, color=color, 
                          edgecolor='black', linewidth=0.5,zorder=3)
                    if value >= 25:
                        add_labels_with_cutoff(rects, ax, 30)
                        

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(
        x_labels,
        fontsize=35,
        rotation=0,
        va='top',
        ha='center',
        position=(0, -0.08), 
        clip_on=False
        )
        ax.tick_params(axis='x', which='major', pad=15)
       
        
        ax.set_xticks(all_x, minor=True)
        ax.set_xticklabels(seq_labels, minor=True,  fontsize=30)
        ax.set_ylabel('Speedup', fontsize=35)
        ax.set_title(title, fontsize=35, pad=20,y=-0.55)
        ax.set_ylim(bottom=0) 
        ax.margins(y=0) 
        ax.margins(x=0.01)
           

        if plot_idx in [0, 1]:
            ax.set_ylim(0, 30)
            ax.set_yticks([0,5,10,15,20,25,30])
            ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
            for y in [10,20]:
                ax.axhline(y=y, color='gray', linestyle=(0, (10, 5)), linewidth=1)
        else:
            ax.set_ylim(0, 30)
            ax.set_yticks([0,5,10,15,20,25,30])
            ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
            for y in [10,20]:
                ax.axhline(y=y, color='gray', linestyle=(0, (10, 5)), linewidth=1)

    label_name=['PyTorch Native', 'MCFuser', 'ByteTransformer', 'FlashAttention2', 'FlexAttention', 'STOF']
    color_keys = ['Torch Naive', 'MCFuser', 'ByteTrans', 'FlashAttn2', 'FlexAttn', 'Our Kernel']
    handles = [plt.Rectangle((0,0), 1,1, fc=colors[key],edgecolor='black', linewidth=0.5) for key in color_keys]
    
    fig.legend(
        handles=handles,
        labels=label_name,
        loc='upper center',
        ncol=6,
        bbox_to_anchor=(0.5, 1.045),
        frameon=False,
        fontsize=35
     
    )
        
    plt.tight_layout()
    plt.savefig('fig10_'+gpu_info+'.pdf', bbox_inches='tight')
    

def main():
    parser = argparse.ArgumentParser(description='Plot MHA performance')
    parser.add_argument('--file_path1', default="../../data/MHA_Performance/Final_sliding_unified4090.txt",)
    parser.add_argument('--file_path2', default="../../data/MHA_Performance/Final_dilated_unified4090.txt",)
    parser.add_argument('--file_path3', default="../../data/MHA_Performance/Final_longformer_unified4090.txt",)
    parser.add_argument('--file_path4', default="../../data/MHA_Performance/Final_bigbird_unified4090.txt",)
    parser.add_argument('--onlymc_file_path', default="../../data/MHA_Performance/attn_perf_only_MCFuser_4090.txt",type=str, help='Path to the input onlymc_txt file.')
    parser.add_argument('--picture_name', default="4090",)
    args = parser.parse_args()

    data_list = []
    for fp in [args.file_path1, args.file_path2, args.file_path3, args.file_path4]:
        raw_data = parse_file(fp, args.onlymc_file_path)
        norm_data = normalize_data(raw_data)
        data_list.append(norm_data)

    subtitles = ['(a) Sliding Window', '(b) Dilated',
                '(c) Longformer', '(d) BigBird']
    
    plot_performance(data_list, subtitles, args.picture_name)

if __name__ == '__main__':
    main()