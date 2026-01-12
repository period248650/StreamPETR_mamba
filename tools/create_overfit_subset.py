#!/usr/bin/env python
"""
创建过拟合测试用的小型数据集 annotation 文件

使用方法：
    cd StreamPETR_mamba_new
    python tools/create_overfit_subset.py

这会从验证集中提取前 N 个样本，创建一个小型 annotation 文件用于过拟合测试。
"""

import pickle
import os

def create_overfit_subset(
    input_pkl='data/nuscenes/nuscenes2d_temporal_infos_val.pkl',
    output_pkl='data/nuscenes/nuscenes2d_temporal_infos_overfit.pkl',
    num_samples=50
):
    """
    从验证集中提取前 N 个样本
    
    Args:
        input_pkl: 输入的 annotation 文件路径
        output_pkl: 输出的小型 annotation 文件路径
        num_samples: 提取的样本数量
    """
    print(f"Loading {input_pkl}...")
    with open(input_pkl, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Original dataset info:")
    print(f"  - Keys: {data.keys()}")
    if 'infos' in data:
        print(f"  - Number of samples: {len(data['infos'])}")
    
    # 提取前 N 个样本
    if 'infos' in data:
        original_count = len(data['infos'])
        data['infos'] = data['infos'][:num_samples]
        print(f"\nExtracted {len(data['infos'])} samples from {original_count}")
    
    # 保存
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    with open(output_pkl, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved to {output_pkl}")
    print(f"\nOverfit subset created successfully!")
    print(f"You can now use this file in your config:")
    print(f"  ann_file=data_root + 'nuscenes2d_temporal_infos_overfit.pkl'")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/nuscenes/nuscenes2d_temporal_infos_val.pkl')
    parser.add_argument('--output', default='data/nuscenes/nuscenes2d_temporal_infos_overfit.pkl')
    parser.add_argument('--num-samples', type=int, default=50)
    args = parser.parse_args()
    
    create_overfit_subset(args.input, args.output, args.num_samples)
