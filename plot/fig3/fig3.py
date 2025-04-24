import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser(description="Generate performance acceleration charts from CSV data.")
parser.add_argument("--input_file", type=str, default="../../data/Motivation/3-moti-4090_3_2.csv", help="Path to the input CSV file")
args = parser.parse_args()


input_file = args.input_file
df = pd.read_csv(input_file)
df.columns = df.columns.str.strip()
operators = df['Operator'].unique()

plt.style.use('default')
fig, axes = plt.subplots(1, len(operators), figsize=(16, 2), sharey=True, gridspec_kw={'wspace': 0})
bar_width = 0.24
colors = ['#afc8ea', '#dff1d7']
y_cutoff = 3
legend_handles = []

def add_labels_with_cutoff(rects, ax, cutoff):
    for rect in rects:
        actual_height = rect.get_height()
        if actual_height > cutoff - 0.01:
            ax.text(rect.get_x() - bar_width / 6,
                    cutoff * 0.98,
                    f'{actual_height:.1f}',
                    ha='right',
                    va='top',
                    fontsize=13,
                    fontname='Times New Roman',
                    color='black')

for idx, op in enumerate(operators):
    op_data = df[df['Operator'] == op].copy()
    op_data = op_data.sort_values(['Batch','SeqLen', 'Hidden'])


    op_data['Triton_Speedup'] = op_data["PyTorch (ms)"] / op_data['Triton (ms)']
    op_data['PyTorch_Baseline'] = 1
    labels = op_data.apply(lambda row: f"({int(row['Batch'])},{int(row['SeqLen'])},{int(row['Hidden'])})", axis=1)
    x = np.arange(len(labels))

    rects1 = axes[idx].bar(x - bar_width/2-0.04, op_data['PyTorch_Baseline'], bar_width,
                           color=colors[0], edgecolor='black', linewidth=1.2)
    rects2 = axes[idx].bar(x + bar_width/2, op_data['Triton_Speedup'], bar_width,
                           color=colors[1], edgecolor='black', linewidth=1.2)

    xtick_positions = x + (- 0.02) 

    axes[idx].set_xticks(xtick_positions)

    axes[idx].set_xticklabels(labels, fontname='Times New Roman', fontsize=100, rotation=45)
    axes[idx].tick_params(axis='x', labelsize=20, pad=2)



    add_labels_with_cutoff(rects2, axes[idx], y_cutoff)

    axes[idx].set_ylim(0, y_cutoff)
    axes[idx].set_yticks(np.arange(0, y_cutoff + 1, 1))
    axes[idx].tick_params(axis='both', labelsize=13)
    axes[idx].set_yticklabels(np.arange(0, y_cutoff + 1, 1), fontname='Times New Roman')


    axes[idx].text(0.5, 0.7, op,fontsize=20, fontname='Times New Roman',
                    ha='center', va='bottom', transform=axes[idx].transAxes)


    axes[idx].grid(True, linestyle='--', alpha=0.6, axis='y')



    if idx == 0:
        axes[idx].set_ylabel('Speedup', fontsize=18, fontname='Times New Roman')
        legend_handles = [rects1[0], rects2[0]]



fig.legend(handles=legend_handles,
           labels=['Detached', 'Fused'],
           prop={'family': 'Times New Roman', 'size': 20},
           loc='upper center',
           ncol=2,
           bbox_to_anchor=(0.513, 1.2),
           frameon=False,
           framealpha=1.0,
           )


input_filename = os.path.basename(input_file) 
output_file = os.path.splitext(input_filename)[0] + '_fig4.pdf' 
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"Chart saved as {output_file}")
