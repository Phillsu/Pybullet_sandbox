# PyBullet 機器人手臂操作

一個基於 PyBullet 的機器人手臂模擬項目，支持鍵盤控制和多種操作模式，包含一鍵重置功能。

## 🚀 項目特色

- 🤖 **完整的機器人手臂物理模擬環境**
- ⌨️ **直觀的鍵盤控制介面**
- 🔄 **兩種控制模式**：關節控制和逆運動學控制
- 🎯 **即時可視化**：目標點和手臂狀態
- ⚡ **基於物理的精確模擬**
- 🔄 **一鍵重置**：快速恢復初始狀態
- 📊 **即時回饋**：關節角度和目標位置顯示

## 📦 安裝依賴

```bash
pip install pybullet numpy
```

## 🎮 快速開始

1. 複製或下載項目文件
2. 運行主程式：
```bash
python robot_arm_simulation.py
```

## 🕹️ 控制說明

### 通用控制
| 按鍵 | 功能 |
|------|------|
| **M** | 切換控制模式 (關節控制/逆運動學控制) |
| **R** | **重置所有關節到初始位置** |
| **Q** | 退出模擬 |

### 關節控制模式
在此模式下，您可以單獨控制每個關節的角度：

| 按鍵 | 功能 |
|------|------|
| **1-7** | 選擇要控制的關節 (1-7) |
| **W** | 增加選中關節的角度 |
| **S** | 減少選中關節的角度 |

### 逆運動學控制模式
在此模式下，您可以直接控制末端執行器的位置，系統會自動計算所需的關節角度：

| 按鍵 | 功能 |
|------|------|
| **I/K** | 沿 X 軸移動目標點 (前進/後退) |
| **J/L** | 沿 Y 軸移動目標點 (左/右) |
| **U/O** | 沿 Z 軸移動目標點 (上/下) |

## 🔧 核心功能

### 一鍵重置功能
項目包含強大的重置功能：
- **即時恢復**：按 `R` 鍵立即重置所有關節
- **完全重置**：關節位置、目標點位置同步重置
- **狀態同步**：內部變數和視覺顯示同時更新
- **操作回饋**：控制台顯示重置確認資訊

```python
def reset_all_joints():
    """重置所有關節到初始位置"""
    for i in range(numJoints):
        p.resetJointState(robotId, i, initialPositions[i])
        p.setJointMotorControl2(robotId, i, p.POSITION_CONTROL, targetPosition=initialPositions[i])
    print("已重置所有關節到初始位置")
```

## 🏗️ 項目結構

```
robot_arm_simulation/
├── robot_arm_simulation.py    # 主程式文件
├── README.md                  # 項目說明文件
└── requirements.txt           # 依賴包列表
```

## 🔬 技術細節

### 使用的機器人模型
項目默認使用 **KUKA LBR iiwa** 7自由度機器人手臂：

- **7個旋轉關節**：完整的運動自由度
- **工業級精度**：適合研究和教育
- **協作機器人**：安全的模擬環境

### 物理引擎配置
- **引擎**：PyBullet 物理引擎
- **重力**：9.8 m/s²
- **模擬頻率**：240Hz
- **碰撞檢測**：即時物理交互

### 控制算法
1. **正向運動學**：關節角度 → 末端位置
2. **逆運動學**：末端位置 → 關節角度
3. **位置控制**：精確的關節伺服控制

## 🚀 擴展功能

### 1. 物體交互
```python
# 添加可抓取的物體
boxId = p.loadURDF("cube.urdf", [0.5, 0, 0.5])
```

### 2. 軌跡規劃
```python
# 實現平滑運動軌跡
waypoints = [[0.5,0,0.5], [0.6,0.1,0.6], [0.4,-0.1,0.7]]
```

### 3. 感測器模擬
```python
# 視覺感測器
camera_view = p.getCameraImage(width=224, height=224)
# 力感測器
contact_points = p.getContactPoints(robotId)
```

### 4. 高級控制
- PID 控制器
- 阻抗控制
- 自適應控制算法

## 🛠️ 開發指南

### 自訂機器人模型
要使用其他機器人模型，修改載入程式碼：
```python
# 替換為您的 URDF 文件路徑
robotId = p.loadURDF("path/to/your/robot.urdf", [0, 0, 0])
```

### 添加新控制模式
在主循環中添加新的控制邏輯：
```python
elif controlMode == "YOUR_NEW_MODE":
    # 實現新的控制邏輯
    pass
```

### 修改重置行為
自訂重置功能：
```python
def custom_reset():
    reset_all_joints()
    # 添加自訂重置邏輯
    custom_initial_position = [0.1, 0.2, -0.1, 0, 0, 0, 0]
    # ...
```

## 🐛 故障排除

### 常見問題

| 問題 | 解決方案 |
|------|----------|
| **顯示問題** | 確保系統支持 OpenGL，嘗試 `p.DIRECT` 模式 |
| **導入錯誤** | 確認安裝所有依賴，檢查 Python 版本 (3.7+) |
| **控制無響應** | 確保視窗焦點，檢查鍵盤映射 |
| **機器人不移動** | 檢查關節限制，確認控制模式 |

### 性能優化
對於較低配置的電腦：
```python
# 降低模擬精度提高性能
p.setPhysicsEngineParameter(fixedTimeStep=1./120.)
```

### 除錯技巧
1. **查看關節資訊**：程式啟動時顯示所有關節詳情
2. **監控控制台輸出**：即時顯示操作回饋
3. **使用重置功能**：遇到問題時按 `R` 重新開始

## 🤝 貢獻指南

我們歡迎各種形式的貢獻！

1. **報告問題**：提交 Issue 描述遇到的問題
2. **功能建議**：提出新的功能想法
3. **程式碼貢獻**：提交 Pull Request
4. **文件改進**：幫助完善文件和範例

### 開發環境設置
```bash
git clone <repository-url>
cd robot_arm_simulation
pip install -r requirements.txt
```

## 🔗 相關資源

- [PyBullet 官方文件](http://pybullet.org/)
- [URDF 模型格式](http://wiki.ros.org/urdf)
- [機器人學導論資源](https://github.com/AtsushiSakai/PythonRobotics)
- [KUKA iiwa 技術文件](https://www.kuka.com/)

*注意：本項目主要用於教育和研究目的。在實際機器人上部署前，請進行充分的安全測試和驗證。*

---
