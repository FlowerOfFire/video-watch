# video-watch

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](./pyproject.toml)
[![Agent Skills](https://img.shields.io/badge/Agent%20Skills-SKILL.md-green.svg)](./SKILL.md)

面向 AI Agent（Antigravity / Claude Code / Codex CLI 等）的轻量视频阅读预处理 CLI 与 **Agent Skill**。

它把任意视频（本地文件、B 站 BV/链接、YouTube 链接）浓缩为低 Token 消耗的图文对齐包：`brief.md` + `grids/*.jpg` 宫格总览图 + 去重后的 `keyframes/*.jpg` + 原语言优先文稿。

## 核心能力

1. **自适应混合拆帧与去重（默认 `--detector auto`）**
   - 第一遍使用 **PySceneDetect**（默认 `ContentDetector(27)` + `StatsManager`）识别真实镜头切换并统计各片段帧间波动。
   - 若某片段属于“高静帧占比 + 局部/小幅跳变”的录屏、PPT 翻页、极简 UI 或竖屏插片，自动仅对该时间窗口补跑局部敏感的 `proxy` 扫描（1fps + `pHash` + 4×4 分块 MAD，并过滤纯黑转场空帧），同时不拆碎实拍运镜与连续口播。
   - 最后执行全局去重，导出高清关键帧及带 `#编号 时间戳` 标注的 `3x3` 宫格总览图；支持 `--ignore-bottom` 与 `--ignore-rect` 屏蔽硬字幕与主播小窗。
2. **原语言优先字幕 + 本地 Qwen3-ASR 兜底**
   - B 站优先拉取原语言/人工字幕轨道（`ai_type=0, type=0`），YouTube 等平台优先原语言及 `*-orig` 轨道，避免优先拿到机翻字幕。
   - 无字幕时自动调用本地 `Qwen3-ASR-1.7B`（按静音停顿 12s–28s 分段，输出带时间戳的 `transcript.md` 与 `transcript.json`）。
3. **图文对齐摘要包 & 单帧放大**
   - 输出 `brief.md`、`transcript.md`、`transcript.json`、`manifest.json`。
   - 支持 `video-watch grab <video> <timestamp>` 按时间戳提取原画质单帧精看。

## 安装与部署

### 1. 安装 CLI（推荐 `uv`）

全局安装为 `video-watch` 命令（源码修改即时生效推荐 `-e`）：

```bash
uv tool install -e .
```

或仅在当前仓库环境同步依赖：

```bash
uv sync
uv run video-watch --help
```

### 2. 安装为 Agent Skill

本仓库遵循 [Agent Skills](https://agentskills.io) 开放规范（根目录包含 [`SKILL.md`](./SKILL.md) 与 [`references/`](./references/)）。

- **从 GitHub 安装**：
  ```bash
  npx skills add FlowerOfFire/video-watch
  ```
- **本地开发链接到全局 Skill 目录（Windows PowerShell 示例）**：
  ```powershell
  # Antigravity 全局发现目录（任选其一，推荐 ~/.gemini/config/skills）
  New-Item -ItemType Junction -Path "$HOME\.gemini\config\skills\video-watch" -Target "$PWD"
  ```

### 外部依赖说明

- **必需**：`ffmpeg` 与 `ffprobe` 在 `PATH` 中可用。
- **可选（按输入源需要）**：
  - B 站下载与原语言字幕：`yutto`、`bilibili-cli`（`bili`）
  - YouTube / 通用视频站：`yt-dlp`
  - 无字幕本地语音转写：独立的 `qwen3-asr` 环境与模型权重（可通过环境变量 `VIDEO_WATCH_QWEN_PYTHON`、`VIDEO_WATCH_QWEN_MODEL`、`VIDEO_WATCH_BILI_PYTHON` 显式指定）。

## 常用命令

```bash
# 1. 默认混合模式处理本地视频
video-watch run "D:\path\to\video.mp4" --out ".\runs\demo"

# 2. 处理 B 站视频（自动抓原语言字幕，无字幕回落 Qwen3-ASR）
video-watch run "BV1xxxxxxxxx" --out ".\runs\bili_demo"

# 3. 屏蔽底部硬字幕与右下角主播小窗，减少误抽帧
video-watch run "lesson.mp4" --ignore-bottom 0.12 --ignore-rect 0.75,0.68,1.0,1.0

# 4. 仅抽帧去重、跳过本地 ASR
video-watch run "video.mp4" --no-asr

# 5. 按时间戳点播高清单帧
video-watch grab "video.mp4" 04:12 --out detail_04m12s.jpg
```

详细参数与输出格式见 [`references/cli-reference.md`](./references/cli-reference.md)。

## 仓库结构

```text
video-watch/
├── SKILL.md                    # Agent Skill 主指令（YAML frontmatter + 工作流）
├── references/
│   └── cli-reference.md        # CLI 全参数、环境变量与输出 schema
├── src/video_watch/
│   ├── __init__.py             # 包入口导出
│   ├── __main__.py             # python -m video_watch 支持
│   ├── watch.py                # CLI 主入口（run / grab）与图文对齐 brief 生成
│   ├── frames.py               # auto 混合分镜、proxy 局部去重、宫格拼图
│   ├── transcript.py           # 原语言字幕检索与 Qwen3-ASR 调度
│   ├── bili_subtitles.py       # B 站字幕轨道排序与拉取
│   └── qwen_asr_chunked.py     # Qwen3-ASR 静音分段转写脚本
├── tests/
│   └── test_video_watch.py     # 离线单元测试
├── pyproject.toml              # uv_build 配置与 video-watch 脚本入口
├── uv.lock
├── LICENSE
└── README.md
```

## 开发测试

```bash
uv sync --group dev
uv run pytest -q
```
