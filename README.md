# KataGo 复盘 Web 插件

基于 KataGo Analysis Engine 的围棋复盘工具，支持 LLM 自然语言解说。

## 功能

- 上传 SGF 棋谱，自动分析每一手
- KataGo 胜率分析 + 候选变化图
- 问题手识别（疑问手 / 恶手）
- LLM 生成自然语言解说（支持 OpenAI / Claude / DeepSeek / Ollama）
- 交互式棋盘 + 胜率曲线

## 快速开始

### 1. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

### 2. 配置

编辑项目根目录下的 `config.yaml`：

- 设置 KataGo 可执行文件与模型权重路径
- 配置 LLM provider 和 API Key

### 3. 下载 KataGo

从 [KataGo Releases](https://github.com/lightvector/KataGo/releases) 下载对应平台的预编译版本，并下载神经网络权重文件（推荐 `kata1-b18c384nbt-s*.bin.gz`）。

### 4. 启动服务

```bash
cd backend
uvicorn main:app --reload
```

浏览器打开 `http://127.0.0.1:8000/static/index.html`。

## 目录结构

```
katago_review/
├── backend/
│   ├── main.py              # FastAPI 入口
│   ├── config.py            # 配置管理
│   ├── sgf_parser.py        # SGF 解析
│   ├── katago_engine.py     # KataGo 引擎封装
│   ├── llm_service.py       # LLM 调用层
│   ├── reviewer.py          # 复盘逻辑
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/
│       ├── app.js
│       ├── board.js
│       └── chart.js
├── config.yaml              # 用户配置
└── README.md
```
