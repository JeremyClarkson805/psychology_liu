# psychology-liu: 刘老师的AI心理问卷分析项目


---

## 1. 项目核心逻辑
1. **人格模拟**：通过 `System Prompt` 为 LLM 设定特定的人生经历、性格特质和情绪倾向。
2. **自动化测验**：系统遍历所有配置的模型与问卷，以 JSON 格式捕获 LLM 的选择结果及背后的“内心感受（ai_reasoning）”。
3. **数据结构化**：结果持久化至 PostgreSQL，包含完整的对话历史与模型原始回复。

---

## 2. 目录结构与模块说明

### 根目录文件
- `问卷1.md` 至 `问卷5.md`: 原始 Markdown 格式问卷（CTQ, ERQ, BRIEF, NSSI 等）。
- `test_countpoint.py`: **评分逻辑校对脚本**。用于离线测试和验证量表计分算法的准确性。
- `debug_data.py`: **数据异常排查工具**。用于检查数据库中答题记录的完整性。
- `runner.lock`: 运行锁，防止多个采集实例冲突。

### `src/` 核心代码
- **`persona_runner.py`**: **项目核心入口**。驱动 LLM 模拟人格并完成所有问卷作答。
- **`db.py`**: **数据持久化层**。包含 PostgreSQL 初始化逻辑及核心表结构：
  - `ai_persona_runs`: 记录每次运行的人格设定与模型。
  - `questionnaire_answers`: 记录具体的题目、选项及 AI 的内心独白。
- **`llm/client.py`**: **通用 LLM 客户端**。支持流式输出、`<think>` 标签过滤及 JSON 提取纠错。
- **`config/llm_config.py`**: **配置中心**。
  - **关键适配**：针对不同模型的 Temperature 限制及 Thinking 模式关闭逻辑。

### `tests/` 目录
- **`tests/ctq_plot.py`**: **核心 CTQ 绘图工具**。专门用于生成多模型在童年创伤各维度上的簇状对比图。

---

## 3. 数据库结构 (Schema)
本项目依赖 PostgreSQL，核心表如下：
- **`ai_persona_runs`**: 存储运行元数据（run_id, persona_prompt, model_name）。
- **`questionnaire_answers`**: 存储答题细节（run_id, questionnaire_name, answer_content, ai_reasoning）。

---

## 4. 安装与运行指南

### 环境准备
```bash
uv sync
```

### 数据库初始化
```bash
# 使用 uv 运行初始化脚本
uv run src/verify_db.py
```

### 核心运行流程
1. **启动数据采集**：
   ```bash
   uv run src/persona_runner.py
   ```
2. **生成核心 CTQ 对比图**：
   ```bash
   uv run tests/ctq_plot.py
   ```
   ```bash
   uv run tests/ctq_plot_separate.py
   ```


