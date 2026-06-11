import MistrasDTA
import os
from numpy.lib.recfunctions import join_by
import numpy as np

# 获取项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))
dta_file = os.path.join(project_root, '0702-2.DTA')

print("=" * 80)
print("DTA 文件数据结构分析")
print("=" * 80)

# 读取数据
rec, wfm = MistrasDTA.read_bin(dta_file)

print(f"\n📊 数据概览:")
print(f"  摘要记录数: {len(rec)}")
print(f"  波形记录数: {len(wfm)}")

# 显示摘要表的列名
print(f"\n摘要表(rec)的字段:")
for i, name in enumerate(rec.dtype.names, 1):
    print(f"  {i}. {name}")

# 显示波形表的列名
print(f"\n波形表(wfm)的字段:")
for i, name in enumerate(wfm.dtype.names, 1):
    print(f"  {i}. {name}")

# 合并数据
merged = join_by(['SSSSSSSS.mmmuuun', 'CH'], rec, wfm)

print(f"\n合并后的记录数: {len(merged)}")

# 显示前5条摘要数据
print(f"\n前5条摘要数据:")
print(f"{'序号':<5} {'时间戳':<15} {'通道':<5} {'能量':<8} {'振幅':<8} {'持续时间':<12}")
print("-" * 60)
for i in range(min(5, len(rec))):
    timestamp = rec['TIMESTAMP'][i]
    ch = rec['CH'][i]
    energy = rec['ENER'][i]
    amp = rec['AMP'][i]
    duration = rec['DURATION'][i]
    print(f"{i+1:<5} {timestamp:<15.2f} {ch:<5} {energy:<8} {amp:<8} {duration:<12}")

# 解释波形分割的原因
print("\n" + "=" * 80)
print("💡 为什么有1986个波形？")
print("=" * 80)
print("""
Mistras AE系统的工作原理：
1. 系统连续监听传感器信号
2. 当检测到一个"击"(hit/event)时，自动提取该事件的波形数据
3. 每个检测到的击都保存为一个单独的摘要记录和对应的波形数据

因此：
✓ 1986个摘要 = 1986个检测到的AE击事件
✓ 1986个波形 = 这些事件对应的完整波形记录
✓ 每个波形都是系统自动分割和采集的结果

这不是人为分割，而是Mistras硬件在实时检测事件时自动执行的。
""")

# 显示波形的具体内容
print("\n前3个波形的详细信息:")
print("-" * 80)
for i in range(min(3, len(merged))):
    print(f"\n波形 #{i+1}:")
    t, V = MistrasDTA.get_waveform_data(merged[i])
    print(f"  事件ID (SSSSSSSS.mmmuuun): {merged[i]['SSSSSSSS.mmmuuun']}")
    print(f"  通道: {merged[i]['CH']}")
    print(f"  采样点数: {len(t)}")
    print(f"  时间范围: {t[0]:.3f} - {t[-1]:.3f} μs")
    print(f"  电压范围: {V.min():.6f} - {V.max():.6f} V")
    print(f"  对应的事件参数:")
    print(f"    - 能量: {merged[i]['ENER']}")
    print(f"    - 振幅: {merged[i]['AMP']}")
    print(f"    - 持续时间: {merged[i]['DURATION']}")
    print(f"    - 上升时间: {merged[i]['RISE']}")

print("\n" + "=" * 80)
print("结论：")
print("=" * 80)
print(f"""
• merged[0] 是第1个检测到的击事件，对应的波形数据
• 之所以有1986个击，是因为在采集期间，Mistras系统检测到了1986个AE事件
• 这是真实的材料损伤或缺陷产生的声发射信号
• 每个波形都包含了该事件从触发到结束的完整信号记录
""")
