
# 情感感知支持聊天机器人


## 项目简介


本项目是一个专注于情感感知的 Web 应用，具备两大核心功能：


- 基于摄像头的实时人脸情绪识别
- 利用 DeepSeek API 结合情绪上下文实现多轮对话



## 当前行为


应用持续分析摄像头画面中的主要人脸情绪，并将检测到的情绪状态实时传递给 DeepSeek 聊天后端。


每次聊天请求会发送：


- 当前人脸情绪标签
- 当前置信度分数
- 最新时间戳
- 重要情绪变化历史
- 多轮对话历史


如果用户持续 5 秒处于愤怒、恐惧、悲伤、厌恶等负面情绪，后端会主动生成安慰性回复。


## 功能特性


- 摄像头人脸情绪检测
- 实时人脸框与概率分布显示
- 重要情绪变化追踪
- DeepSeek 多轮对话
- 持续负面情绪自动安慰
- 对话与情绪历史持久化
- 运行日志与会话快照


## 技术栈


- Python
- Flask
- TensorFlow / Keras
- OpenCV
- Dlib
- NumPy
- SciPy
- OpenAI Python SDK（用于 DeepSeek API）
- HTML / CSS / JavaScript


## 项目结构

```text
emotion_detection_using_chatbot/
├── README.md
├── ui/
│   ├── app.py
│   ├── requirements.txt
│   ├── dlib-19.24.1-cp311-cp311-win_amd64.whl
│   ├── templates/
│   │   └── index.html
│   ├── captures/
│   └── runtime_data/
└── venv/
```


说明：


- `ui/captures/` 运行时自动创建，用于保存截图。
- `ui/runtime_data/` 运行时自动创建，用于保存会话状态和日志。
- 人脸关键点模型文件需本地自备，不建议上传 GitHub。


## 本地所需文件


`ui/Models/` 目录下需放置：


- `shape_predictor_68_face_landmarks.dat`


缺少该文件则无法进行人脸关键点检测。


情绪识别模型已切换为 Hugging Face 公共模型：`dima806/facial_emotions_image_detection`


## 环境变量


请勿将 API 密钥等敏感信息写入源码。


请参考 `.env.example` 在项目根目录创建 `.env` 文件。


示例 `.env`：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```


应用会自动加载根目录或 `ui/` 下的 `.env` 文件。


运行前需设置如下环境变量：

```powershell
$env:DEEPSEEK_API_KEY="your_token_here"
```


或：

```powershell
$env:OPENAI_API_KEY="your_token_here"
```


可选配置：

```powershell
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_MODEL="deepseek-chat"
$env:HF_EMOTION_MODEL="dima806/facial_emotions_image_detection"
$env:CAMERA_INDEX="0"
$env:CAMERA_BACKEND="auto"
```


摄像头说明：

- `CAMERA_INDEX` 默认为 `0`，如摄像头为其他编号可尝试 `1`、`2`。
- `CAMERA_BACKEND` 可选 `auto`、`default`、`dshow`、`msmf`（Windows）。
- `/start_camera` 和 `/system_status` 接口会返回当前摄像头参数，便于诊断。


## 安装方法


在项目根目录执行：

```powershell
pip install -r ui/requirements.txt
```


推荐 Windows 环境下：

```powershell
python -m venv venv
.\venv\Scripts\Activate
python -m pip install --upgrade pip
pip install -r ui/requirements.txt
```


如 Dlib 安装失败，可用 `ui/` 目录下的 whl 文件手动安装（需 Python 版本匹配）：

Example:

```powershell
pip install ui/dlib-19.24.1-cp311-cp311-win_amd64.whl
```


## 运行方式


在项目根目录执行：

```powershell
python ui/app.py
```


如需指定虚拟环境：

```powershell
d:/emotion_detection_using_chatbot/venv/Scripts/python.exe ui/app.py
```


浏览器访问：

```text
http://127.0.0.1:5000/
```


## 运行时输出


运行时会生成：


- `ui/runtime_data/activity_log.jsonl`：对话、情绪变化、错误日志
- `ui/runtime_data/session_state.json`：最新会话与情绪快照
- `ui/captures/`：截图与情绪数据


## 已移除模块


以下模块已不再包含在主流程中：


- 文本情感分析及相关预处理
- 音频情绪识别及相关界面
- 本地 Llama 测试脚本与本地 LLM 聊天
- 多模态融合结果面板


## 注意事项


- 当前 Python 环境需安装 `ui/requirements.txt` 中所有依赖。
- 聊天功能需提前配置好 DeepSeek API 密钥。
- 摄像头情绪识别依赖本地 68 点人脸关键点模型和 Hugging Face 情绪模型。


## 许可协议


本仓库仅供学术、研究与实验开发用途。






