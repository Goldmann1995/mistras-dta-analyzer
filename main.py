import MistrasDTA
from numpy.lib.recfunctions import join_by
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 获取项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))

# 读取 DTA 文件
dta_file = os.path.join(project_root, '0702-2.DTA')

if os.path.exists(dta_file):
    print(f"正在读取 {dta_file}...")
    
    # 读取摘要表和波形数据（完整数据）
    print("\n=== 读取数据 ===")
    rec, wfm = MistrasDTA.read_bin(dta_file)
    print(f"读取到 {len(rec)} 条 AE 摘要")
    print(f"读取到 {len(wfm)} 条波形数据")
    
    # 合并摘要和波形表
    if len(wfm) > 0:
        merged = join_by(['SSSSSSSS.mmmuuun', 'CH'], rec, wfm)
        print(f"合并后共有 {len(merged)} 条记录")
        
        # 创建第一个可视化（原有的四个子图）
        fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
        fig1.suptitle('MistrasDTA 波形和统计数据可视化', fontsize=16, fontweight='bold')
        
        # ===== 1. 绘制单个波形 =====
        print("\n绘制单个波形...")
        ax1 = axes1[0, 0]
        t, V = MistrasDTA.get_waveform_data(merged[0])
        ax1.plot(t, V, 'b-', linewidth=0.8)
        ax1.set_xlabel('时间 (μs)', fontsize=10)
        ax1.set_ylabel('电压 (V)', fontsize=10)
        ax1.set_title('第1个波形', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # ===== 2. 绘制多个波形叠加 =====
        print("绘制多个波形叠加...")
        ax2 = axes1[0, 1]
        num_waves = min(20, len(merged))  # 最多显示20个波形
        for i in range(num_waves):
            t_i, V_i = MistrasDTA.get_waveform_data(merged[i])
            ax2.plot(t_i, V_i, alpha=0.4, linewidth=0.6)
        ax2.set_xlabel('时间 (μs)', fontsize=10)
        ax2.set_ylabel('电压 (V)', fontsize=10)
        ax2.set_title(f'前{num_waves}个波形叠加', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # ===== 3. 能量分布直方图 =====
        print("绘制能量分布...")
        ax3 = axes1[1, 0]
        energies = rec['ENER']
        ax3.hist(energies, bins=50, color='green', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('能量', fontsize=10)
        ax3.set_ylabel('频数', fontsize=10)
        ax3.set_title('能量分布直方图', fontsize=11, fontweight='bold')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # ===== 4. 振幅 vs 能量散点图 =====
        print("绘制振幅与能量关系...")
        ax4 = axes1[1, 1]
        amplitudes = rec['AMP']
        scatter = ax4.scatter(amplitudes, energies, alpha=0.5, s=20, c=rec['DURATION'], cmap='viridis')
        ax4.set_xlabel('振幅', fontsize=10)
        ax4.set_ylabel('能量', fontsize=10)
        ax4.set_title('振幅 vs 能量 (色彩=持续时间)', fontsize=11, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('持续时间', fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存第一个图表
        output_file1 = os.path.join(project_root, 'waveform_visualization.png')
        plt.savefig(output_file1, dpi=150, bbox_inches='tight')
        print(f"\n✓ 第一个可视化已保存至: {output_file1}")
        plt.close()
        
        # ==================== 创建第二个可视化：通道分析 ====================
        print("\n=== 生成通道分析图表 ===")
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
        fig2.suptitle('通道分析', fontsize=16, fontweight='bold')
        
        # ===== 1. 所有通道的撞击次数 =====
        print("绘制所有通道的撞击次数...")
        ax_channels = axes2[0]
        channels = rec['CH']
        unique_channels, counts = np.unique(channels, return_counts=True)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_channels)))
        bars = ax_channels.bar(unique_channels, counts, color=colors, edgecolor='black', linewidth=1.5)
        
        # 在柱子顶部添加数值
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax_channels.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(count)}',
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax_channels.set_xlabel('通道号', fontsize=11, fontweight='bold')
        ax_channels.set_ylabel('撞击次数', fontsize=11, fontweight='bold')
        ax_channels.set_title('各通道撞击次数', fontsize=12, fontweight='bold')
        ax_channels.grid(True, alpha=0.3, axis='y')
        ax_channels.set_xticks(unique_channels)
        
        # ===== 2. 通道2的幅值 vs 时间 =====
        print("绘制通道2的幅值 vs 时间...")
        ax_ch2 = axes2[1]
        
        # 筛选通道2的数据
        ch2_mask = rec['CH'] == 2
        ch2_rec = rec[ch2_mask]
        
        if len(ch2_rec) > 0:
            # 获取时间戳并转换为相对时间（秒）
            timestamps = ch2_rec['TIMESTAMP']
            start_time = timestamps.min()
            relative_time = (timestamps - start_time)  # 秒为单位
            
            # 获取原始幅值（直接用，不转换为dB）
            amplitudes_ch2 = ch2_rec['AMP']
            
            # 绘制散点图
            scatter2 = ax_ch2.scatter(relative_time, amplitudes_ch2, alpha=0.6, s=30, c=relative_time, cmap='viridis')
            
            ax_ch2.set_xlabel('时间 (秒)', fontsize=11, fontweight='bold')
            ax_ch2.set_ylabel('幅值', fontsize=11, fontweight='bold')
            ax_ch2.set_title('通道2：幅值 vs 时间', fontsize=12, fontweight='bold')
            ax_ch2.grid(True, alpha=0.3)
            
            cbar2 = plt.colorbar(scatter2, ax=ax_ch2)
            cbar2.set_label('时间 (秒)', fontsize=10)
            
            # 统计信息
            print(f"\n通道2统计信息:")
            print(f"  撞击次数: {len(ch2_rec)}")
            print(f"  时间范围: {relative_time.min():.2f} - {relative_time.max():.2f} 秒")
            print(f"  幅值范围: {amplitudes_ch2.min()} - {amplitudes_ch2.max()}")
            print(f"  幅值平均值: {amplitudes_ch2.mean():.4f}")
            print(f"  幅值标准差: {amplitudes_ch2.std():.4f}")
            
            # 添加通道6的统计信息
            ch6_mask = rec['CH'] == 6
            ch6_rec = rec[ch6_mask]
            if len(ch6_rec) > 0:
                amplitudes_ch6 = ch6_rec['AMP']
                print(f"\n通道6统计信息:")
                print(f"  撞击次数: {len(ch6_rec)}")
                print(f"  幅值范围: {amplitudes_ch6.min()} - {amplitudes_ch6.max()}")
                print(f"  幅值平均值: {amplitudes_ch6.mean():.4f}")
                print(f"  幅值标准差: {amplitudes_ch6.std():.4f}")
        else:
            ax_ch2.text(0.5, 0.5, '通道2无数据', ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        
        # 保存第二个图表
        output_file2 = os.path.join(project_root, 'channel_analysis.png')
        plt.savefig(output_file2, dpi=150, bbox_inches='tight')
        print(f"\n✓ 通道分析图表已保存至: {output_file2}")
        plt.close()
        
        # 打印统计信息
        print("\n=== 全局统计信息 ===")
        print(f"能量: 最小={energies.min()}, 最大={energies.max()}, 平均={energies.mean():.2f}")
        print(f"振幅: 最小={amplitudes.min()}, 最大={amplitudes.max()}, 平均={amplitudes.mean():.2f}")
        print(f"持续时间: 最小={rec['DURATION'].min()}, 最大={rec['DURATION'].max()}, 平均={rec['DURATION'].mean():.2f}")
else:
    print(f"错误: 找不到文件 {dta_file}")
