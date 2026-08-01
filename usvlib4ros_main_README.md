# usvlib4ros2 运行指南

> 本文档面向测试人员，说明如何运行 `main.py` 连接仿真平台

---

## 一、环境要求

### 1.1 软件环境

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| Python | 3.10+ | 建议使用 Anaconda/Miniconda 管理环境 |
| roslibpy | 1.6.0 | 连接 ROS2 的 Python 库 |
| rosbags | 0.9.23 | ROS bag 文件处理库 |

### 1.2 依赖安装

项目依赖包及版本（见 `{项目根目录}\requirements.txt`）：

#### 核心依赖（必须安装）

| 包名 | 版本 | 说明 |
|------|------|------|
| roslibpy | 1.6.0 | 连接 ROS2/rosbridge 的 WebSocket 客户端 |
| rosbags | 0.9.23 | ROS bag 文件处理库 |

```bash
# 安装核心依赖
pip install roslibpy==1.6.0 rosbags==0.9.23
```

#### DQN 模块依赖（可选，用于强化学习导航）

| 包名 | 版本 | 说明 |
|------|------|------|
| torch | CPU 版本 | PyTorch 深度学习框架 |
| numpy | 最新版 | 数值计算库 |

```bash
# 安装 PyTorch CPU 版本（清华镜像）
pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 numpy
pip install numpy
```

#### 标准库依赖（Python 自带，无需安装）

- `time`, `sys`, `os`, `json`, `threading`, `logging`
- `datetime`, `math`, `typing`, `collections`

### 1.3 前置条件

运行 `main.py` 前需确保以下环境已就绪：

1. **Ubuntu VM 已启动**并运行 ros2-image 容器
2. **EmboUnity.exe 已启动**并进入场景
3. **MATLAB Simulink 模型已点击运行**
4. **ROSbridge 服务**监听端口 9090

### 1.4 项目目录说明

本文档使用 `{项目根目录}` 表示 usvlib4ros2 项目的位置，**请根据实际情况替换**。

> **示例**：
> - 您的项目目录：`D:\MyProject\usvlib4ros2`
> - 配置文件：`{项目根目录}\usvlib4ros\main.py` → `D:\MyProject\usvlib4ros2\usvlib4ros\main.py`
> - 依赖文件：`{项目根目录}\requirements.txt` → `D:\MyProject\usvlib4ros2\requirements.txt`

---

## 二、项目架构

网盘目录 `智能导航C4-2026` 包含以下组件：

| 文件夹 | 说明 |
|--------|------|
| **model** | MATLAB/Simulink 导航控制模型（含 README 说明文档） |
| **NavAlg** | 算法示例 Demo（如 DQN 强化学习导航） |
| **ROS** | Ubuntu 虚拟机镜像，运行 ros2-image 容器 |
| **usvlib4ros** | 本项目代码，Python 连接 ROS 的客户端库 |
| **具身智能仿真平台** | EmboUnity.exe 仿真界面环境 |

> **提示**：运行 `main.py` 前需确保 ROS 虚拟机和 EmboUnity 已启动。

---

## 三、快速开始

### 3.1 启动步骤

按照以下顺序启动各组件：

#### 步骤1：启动虚拟机

1. 启动 Ubuntu VM 镜像
2. **等待 1 分钟**，确保 ros2-image 容器完全启动

#### 步骤2：配置 EmboUnity

1. 打开 `具身智能仿真平台\EmboUnity.exe`
2. 进行相关配置（详见 [具身智能仿真平台配置文档](https://spaitlab.github.io/Maritime-Intelligent-Navigation-2025/操作手册/环境搭建.html)）

> **配置参考**：[https://spaitlab.github.io/Maritime-Intelligent-Navigation-2025/](https://spaitlab.github.io/Maritime-Intelligent-Navigation-2025/) → 操作手册 → 环境搭建

#### 步骤3：启动 Simulink 模型

1. 打开 MATLAB
2. 打开无人船 Simulink 模型（位于 `model\C4_2026` 目录）
3. 在 MATLAB 中执行启动脚本：

```matlab
cd Model\C4_2026
start_simulation
```


#### 步骤4：运行 main.py

```bash
cd {项目根目录}
python usvlib4ros/main.py
```

### 3.2 安装依赖

```bash
# 进入项目根目录（请根据实际情况替换）
cd {项目根目录}

# 创建虚拟环境（推荐）
conda create -n usvtest python=3.10
conda activate usvtest

# 安装项目依赖（版本见 1.2 节）
pip install -r requirements.txt
```

> **提示**：如果项目目录包含中文或特殊字符，建议使用英文目录名。

### 3.3 运行 main.py

```bash
# 方法1：直接运行（使用默认配置）
python {项目根目录}/usvlib4ros/main.py

# 方法2：指定配置
python {项目根目录}/usvlib4ros/main.py --host 192.168.213.132 --device-id ID_09e4063515d81b2f7352e15bdd53294ace675e96
```

---

## 四、配置说明

### 3.1 参数配置

在 `main.py` 的 `USVNavMain.start()` 调用中修改参数：

```python
USVNavMain.start(
    host="192.168.213.132",      # 虚拟机IP地址
    port=9090,                     # rosbridge端口（固定不变）
    deviceId="ID_09e4063515d81b2f7352e15bdd53294ace675e96",  # 设备唯一标识
    enable_debug=True,             # 是否启用调试输出
    laser_debug_freq=0.5,          # 激光雷达打印频率（Hz）
    device_debug_freq=1.0,         # 设备状态打印频率（Hz）
    control_debug_freq=2.0         # 控制指令打印频率（Hz）
)
```

### 3.2 参数说明

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `host` | Ubuntu 虚拟机 IP 地址 | 必须配置 | `"192.168.213.132"` |
| `port` | rosbridge WebSocket 端口 | `9090` | `9090` |
| `deviceId` | EmboUnity 中智能体的唯一标识 | 必须配置 | `"ID_09e4063515d81b2f7352e15bdd53294ace675e96"` |
| `enable_debug` | 是否输出调试日志 | `True` | `True` / `False` |
| `laser_debug_freq` | 激光雷达日志打印频率 | `0.5` Hz | `0.5` = 每2秒打印一次 |
| `device_debug_freq` | 设备状态日志打印频率 | `1.0` Hz | `1.0` = 每秒打印一次 |
| `control_debug_freq` | 控制指令日志打印频率 | `2.0` Hz | `2.0` = 每0.5秒打印一次 |

### 3.3 如何获取配置值

#### 获取虚拟机 IP

在 Ubuntu VM 终端执行：

```bash
hostname -I
```

或

```bash
ip addr show
```

输出示例：`192.168.213.132`

#### 获取 deviceId

1. 启动 EmboUnity.exe
2. 登录后选择智能体，deviceId 会显示在界面或配置文件中
3. 示例值：`ID_09e4063515d81b2f7352e15bdd53294ace675e96`

---

## 五、运行示例

### 5.1 正常启动输出

成功连接后，终端应显示：

```
[Main] USV Navigation Service started with debug logging enabled.
[Main] Press Ctrl+C to stop.

[ROS Debug] Connected to rosbridge at 192.168.213.132:9090
[ROS DEBUG] [激光雷达数据] Topic: usv/ID_xxx/laser/scan, angle_min: -3.14 ...
[ROS DEBUG] [设备状态] latitude: 30.xxxx, longitude: 120.xxxx, heading: 45.5°
[ROS DEBUG] [控制指令] linear.x: 0.5, angular.z: 0.0
```

### 5.2 常见错误及处理

| 错误信息 | 可能原因 | 解决方法 |
|---------|---------|---------|
| `Connection refused` | VM 未启动或 rosbridge 未运行 | 检查 VM 状态，确认 ros2-image 容器运行 |
| `Timeout error` | 网络不通或防火墙拦截 | 检查宿主机与 VM 网络连接，关闭防火墙 |
| `Invalid deviceId` | deviceId 与 EmboUnity 中不匹配 | 确认 deviceId 大小写正确 |
| `No data received` | EmboUnity 未启动或未进入场景 | 确认 EmboUnity 已启动并选择智能体 |

---

## 六、调试功能

### 6.1 关闭调试输出

如不需要频繁的日志输出，可关闭调试：

```python
USVNavMain.start(
    host="192.168.213.132",
    port=9090,
    deviceId="ID_xxx",
    enable_debug=False      # 关闭调试输出
)
```

### 6.2 调整打印频率

根据需要调整各话题的打印频率：

```python
# 高频率详细调试
laser_debug_freq=2.0    # 每0.5秒打印一次激光雷达数据
device_debug_freq=2.0    # 每0.5秒打印一次设备状态
control_debug_freq=5.0  # 每0.2秒打印一次控制指令

# 低频率减少输出
laser_debug_freq=0.1     # 每10秒打印一次
device_debug_freq=0.5    # 每2秒打印一次
control_debug_freq=0.5   # 每2秒打印一次
```

### 6.3 查看实时统计数据

按 `Ctrl+C` 停止服务后，会打印统计数据：

```
[Main] Stopping USV Navigation Service...
[Debug Stats]
  Total laser scans received: 1256
  Total device status updates: 628
  Total control commands sent: 2512
```

---

## 七、测试检查清单

运行 `main.py` 后，按照以下清单验证功能：

| 检查项 | 预期结果 | 实际结果 |
|--------|---------|---------|
| 1. 连接成功 | 显示 `[ROS Debug] Connected to rosbridge` | |
| 2. 激光雷达数据 | 每2秒输出激光雷达日志 | |
| 3. 设备状态数据 | 每秒输出经纬度、航向角、速度 | |
| 4. 控制指令 | 每0.5秒输出控制指令日志 | |
| 5. EmboUnity 联动 | Unity 场景中无人船响应控制 | |
| 6. 正常停止 | Ctrl+C 后显示统计数据 | |

---

## 八、常见问题

### Q1: 出现 `ModuleNotFoundError: No module named 'usvlib4ros'`

**原因**：未将项目目录加入 Python 路径

**解决**：
```bash
# 在项目根目录执行（Windows PowerShell）
$env:PYTHONPATH = "$env:PYTHONPATH;{项目根目录}\usvlib4ros"
python {项目根目录}\usvlib4ros\main.py

# 或在 Python 代码中临时添加
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'usvlib4ros'))
```

### Q2: 出现 `WebSocket connection failed`

**原因**：rosbridge 未启动或端口错误

**解决**：
1. 确认 VM 中 ros2-image 容器运行：`docker ps | grep ros2-image`
2. 确认 rosbridge 监听端口：`docker exec ros2-image ss -tlnp | grep 9090`
3. 检查 host 参数是否正确

### Q3: 数据一直为 0 或不变

**原因**：EmboUnity 未正确启动或 MATLAB 未运行

**解决**：
1. 确认 EmboUnity 已进入场景并选择智能体
2. 确认 MATLAB 中 Simulink 模型正在运行（状态为 Running）

### Q4: 停止后程序不退出的处理

有时 `Ctrl+C` 可能无法立即停止，可再次按 `Ctrl+C` 或关闭终端窗口

---

## 九、联系支持

如遇到本文档未覆盖的问题，请联系开发人员并提供：
1. 完整的终端输出日志
2. 项目根目录路径（{项目根目录}）
3. VM IP 和 deviceId 配置
4. 测试环境描述（哪个步骤出错）

---

*文档版本：v1.0*
*更新日期：2026-05-15*