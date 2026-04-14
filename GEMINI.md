# GEMINI.md - 项目指令与上下文

## 项目概述
**项目名称**: psychology-liu (自伤行为风险评估问卷相似度分析系统)
**核心目标**: 利用大语言模型（LLM）模拟具有特定心理背景的“人格”（Personas），自动化回答标准化的自伤行为及相关心理评估问卷。通过收集不同模型、不同人格下的回答数据，进行后续的相似度分析和风险评估研究。

## 技术架构
- **语言**: Python 3.12+ (管理工具: `uv`)
- **LLM 交互**: 封装了 `OpenAI` SDK 的 `LLMClient`，支持流式输出、自动重试以及针对不同模型（Qwen, DeepSeek, Kimi, Gemini, GPT等）的参数适配（如 Temperature 限制、Thinking 模式关闭）。
- **数据存储**: PostgreSQL。表结构包含 `ai_persona_runs`（运行记录）和 `questionnaire_answers`（答题明细）。
- **问卷格式**: 存储在根目录下的 `问卷1.md` 至 `问卷5.md`，支持表格和列表两种解析方式。
- **配置管理**: 使用 `Pydantic` 进行类型安全的配置管理（`LLMConfig`）。

## 关键模块
- `src/persona_runner.py`: **核心入口**。定义人格设定，驱动 LLM 遍历问卷进行答题，并将过程日志及最终结果存入数据库。
- `src/llm/client.py`: 通用 LLM 客户端，处理复杂的 API 调用逻辑和结果清理（剔除 `<think>` 标签）。
- `src/questionnaire_parser.py`: 问卷解析器，将 Markdown 文本转换为结构化题目数据。
- `src/db.py`: 数据库交互层，负责初始化表结构及读写操作。
- `src/config/llm_config.py`: 存放 API 密钥、模型列表及推理参数。

## 运行与开发
### 1. 环境准备
```bash
# 使用 uv 同步依赖
uv sync
```
*注：需配置好本地 PostgreSQL 数据库，并在 `src/db.py` 中检查连接配置。*

### 2. 数据库初始化
目前项目中可通过 `python src/verify_db.py` 或直接调用 `src/db.py` 中的 `init_db()` 函数进行初始化。

### 3. 测试与运行
- **测试 LLM 连接**: `python tests/test_llm.py`
- **启动模拟答题**: `python src/persona_runner.py`

## 开发约定
1. **人格设定**: 在 `src/persona_runner.py` 的 `PERSONAS` 列表中添加新的人格描述。
2. **问卷更新**: 直接修改根目录下的 Markdown 文件，确保遵循表格或带编号的列表格式。
3. **模型添加**: 在 `src/config/llm_config.py` 的 `model_names` 中添加新模型，并根据需要在 `TEMPERATURE_ONE_MODELS` 或 `THINKING_MODELS` 中定义特殊适配。
4. **异常处理**: `LLMClient` 已具备基本的重试机制，业务逻辑层应优先通过数据库事务确保数据一致性。

## 待办事项 (TODO)
- [ ] 实现结果的相似度分析算法（基于 NumPy/SciPy）。
- [ ] 增加更多样化的人格设定。
- [ ] 提供更便捷的 CLI 工具用于初始化数据库和导出报告。
