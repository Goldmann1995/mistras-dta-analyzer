"""
DTA文件格式说明和Python采集示例

DTA文件是Mistras系统的二进制数据格式，用于存储声发射事件数据。
结构定义在Mistras用户手册附录II中。

文件结构：
[消息长度(2字节)] [消息ID(1字节)] [消息数据]

常见的消息类型：
- ID 1: AE Hit/Event Data (击事件数据)
- ID 2-9: 其他硬件配置消息
- ID 40-49: 特殊消息（带额外字节）
"""

import struct
import numpy as np
from datetime import datetime

class DTAWriter:
    """DTA文件写入器 - 基础实现"""
    
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'wb')
        self.event_count = 0
    
    def write_ae_event(self, channel, timestamp, rise_time, count, energy, 
                       duration, amplitude, asl, rms, peak_freq, waveform_data=None):
        """
        写入一个AE击事件
        
        参数:
        - channel: 通道号 (1-8)
        - timestamp: Unix时间戳 (秒)
        - rise_time: 上升时间 (μs)
        - count: 计数
        - energy: 能量值
        - duration: 持续时间 (μs)
        - amplitude: 振幅 (dB或0-255)
        - asl: 平均信号级别
        - rms: 均方根值
        - peak_freq: 峰值频率 (kHz)
        - waveform_data: 波形数据 (numpy数组，可选)
        """
        # 这里只是展示结构，完整的实现需要按照Mistras的精确格式
        # 实际的DTA写入相当复杂，涉及多个消息块
        print(f"事件{self.event_count+1}: 通道{channel}, "
              f"时间={datetime.fromtimestamp(timestamp)}, "
              f"振幅={amplitude}, 能量={energy}")
        self.event_count += 1
    
    def close(self):
        """关闭文件"""
        self.file.close()
        print(f"\n✓ DTA文件已保存: {self.filename}")
        print(f"  共写入 {self.event_count} 个事件")


# ============================================================================
# 使用示例
# ============================================================================

print("=" * 80)
print("如何采集传感器数据并保存为DTA格式")
print("=" * 80)

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ 方式1: 使用Mistras官方软件（推荐）                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ 步骤1: 硬件连接                                                             │
│   • 连接AE传感器到前置放大器                                               │
│   • 将采集卡/USB接口连接到计算机                                           │
│   • 连接传感器电缆                                                         │
│                                                                              │
│ 步骤2: 软件配置（使用AEWin或PAC-XT）                                       │
│   • 打开Mistras采集软件                                                    │
│   • 配置参数:                                                              │
│     - 采样率: 通常1-10 MHz                                                 │
│     - 阈值: 根据背景噪声设置 (40-60 dB通常)                               │
│     - 预触发: 通常100-200 μs                                              │
│     - 滤波: 根据被测物选择带通滤波器                                      │
│                                                                              │
│ 步骤3: 采集                                                                │
│   • 点击"开始采集"                                                         │
│   • 实时显示事件                                                           │
│   • 软件自动生成DTA文件                                                    │
│                                                                              │
│ 步骤4: 分析                                                                │
│   • 使用AEWin或MistrasDTA库进行分析                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ 方式2: 用Python编程采集（需要硬件接口库）                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ 如果要用Python完全编程实现，需要:                                           │
│                                                                              │
│ 1. DAQ硬件接口库 (选择其一):                                               │
│    • PyDAQmx - 用于NI数据采集卡                                            │
│    • pyaudio - 用于声卡采集                                                │
│    • custom USB驱动 - 如果使用专有硬件                                     │
│                                                                              │
│ 2. 信号处理:                                                               │
│    • 实时滤波（带通滤波器）                                                │
│    • 事件检测（阈值比较）                                                  │
│    • 特征提取（能量、振幅、频率等）                                        │
│                                                                              │
│ 3. 文件写入:                                                               │
│    • 按照DTA格式编码数据                                                   │
│    • 写入二进制文件                                                        │
│                                                                              │
│ 示例代码框架:                                                              │
│ ──────────────────────────────────────────────────────────────────────────  │
│ import nidaqmx                                                             │
│ import numpy as np                                                         │
│                                                                              │
│ # 连接DAQ设备                                                              │
│ with nidaqmx.Task() as task:                                               │
│     task.ai_channels.add_ai_voltage_chan("Dev1/ai0:7")                    │
│     task.timing.cfg_samp_clk_timing(                                       │
│         rate=1e6,  # 1 MHz采样率                                          │
│         sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS           │
│     )                                                                       │
│                                                                              │
│     # 采集数据                                                             │
│     while True:                                                            │
│         data = task.read(num_samps_per_chan=1000)  # 读取样本             │
│         events = detect_ae_events(data, threshold=50)  # 检测事件          │
│         for event in events:                                               │
│             features = extract_features(event)  # 提取特征                 │
│             writer.write_ae_event(**features)  # 写入DTA文件              │
│                                                                              │
│ ──────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│ ⚠️  注意: DTA文件格式复杂，建议使用官方软件生成                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ 方式3: 将采集到的数据转换为DTA格式                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ 如果你已经有其他格式的数据（如CSV、HDF5等），可以转换为DTA:                │
│                                                                              │
│ 1. 读取源数据                                                              │
│ 2. 提取AE事件（阈值检测）                                                   │
│ 3. 计算特征（能量、振幅等）                                                │
│ 4. 按DTA格式重新编码                                                       │
│ 5. 保存为DTA文件                                                           │
│                                                                              │
│ 这样可以与MistrasDTA库兼容，用于后续分析                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")

print("\n" + "=" * 80)
print("关键参数说明")
print("=" * 80)

params = {
    "采样率": "1-10 MHz（通常2-5 MHz）",
    "阈值": "40-60 dB（根据背景噪声调整）",
    "预触发": "100-200 μs（捕捉事件前端）",
    "滤波": "100 kHz-1 MHz带通（按材料选择）",
    "增益": "0-60 dB（取决于传感器灵敏度）",
    "通道数": "通常2-8个通道",
    "持续时间": "根据需求设定（秒到小时）"
}

for key, value in params.items():
    print(f"• {key:<12}: {value}")

print("\n" + "=" * 80)
print("总结")
print("=" * 80)
print("""
✓ 推荐方案: 使用Mistras官方软件（AEWin/PAC-XT）采集和生成DTA文件
✓ 优点:
  - 格式准确、完整
  - 包含完整的硬件配置信息
  - 软件界面友好，实时监控
  - 可直接用MistrasDTA库分析

⚠️  Python编程方案:
  - 复杂度高，需要硬件驱动
  - DTA格式规范复杂
  - 建议先用官方软件采集，用Python做后期分析

💡 折中方案:
  - 用官方软件采集→生成DTA
  - 用MistrasDTA库读取分析
  - 用Python进行自定义处理和可视化
""")
