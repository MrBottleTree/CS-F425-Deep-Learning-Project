# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CS F425 Deep Learning course project. The goal is to fine-tune a language model (QLoRA on Phi-2 or similar, following `Lab02_QLoRA_NL2SQL_fixed.ipynb`) to act as an **AI agent for structured data analysis**. Given a natural language query, the agent must output a JSON object with a `"actions"` list and an `"answer"` field.

**Deadline: April 16, 11:59 AM (Noon)**. Training must run on free-tier GPU (Google Colab / Kaggle) within 3–6 hours.

## Key Constraint

Do **not** modify `tool_executor.py` or the tool implementations. Prompt design and fine-tuning strategy are the only permitted modifications to the pipeline.

## Data & Files

| File | Role |
|------|------|
| `sales_data.csv` | Training sales data — columns: `date, year, month, city, region, product, category, revenue, units_sold, cost, profit` |
| `agent_trajectories_2k.json` | 2000 training examples: `{"query": "...", "actions": [...]}` — no `"answer"` field, only the action sequence |
| `tool_executor.py` | `ToolExecutor` class — takes internal dict-format actions, executes on a DataFrame |
| `run_pipeline.py` | Parses string-format actions → internal dict format → `ToolExecutor.execute()` → writes results |
| `example_use_tool_executor.py` | Shows how to call `ToolExecutor` directly |

## Tool API (Strict Syntax)

The agent must output actions using **exactly** this string syntax:

```
filter_data(column='col', value=val)
group_by(column='col')
aggregate_sum(column='col')
aggregate_mean(column='col')
aggregate_count(column='col')    # likely supported
sort_by(column='col', order='desc')   # or 'asc'
top_k(k=N)
```

`run_pipeline.py:parse_agent_action()` maps these strings to the internal `{"tool": ..., "args": {...}}` dicts that `ToolExecutor` consumes. The `ToolExecutor` supports ops: `filter`, `groupby`, `aggregate`, `sort`, `topk`.

## Required Agent Output Format

```json
{
  "actions": ["filter_data(column='year', value=2022)", "aggregate_sum(column='revenue')"],
  "answer": 52345678.12
}
```

No text outside the JSON. The `answer` field must be the actual computed result after executing the actions against the CSV.

## Fine-Tuning Approach (from Lab02 reference)

The reference notebook (`Lab02_QLoRA_NL2SQL_fixed.ipynb`) demonstrates:

1. **Model**: `microsoft/phi-2` loaded in 4-bit (NF4) via `bitsandbytes` + `BitsAndBytesConfig`
2. **LoRA config**: `r=16`, `alpha=32`, `dropout=0.05` applied to attention projection layers
3. **Training**: `SFTTrainer` from `trl`, `num_train_epochs=2`, ~2000 examples on T4 GPU
4. **Prompt template**: Structured `### Task / ### Schema / ### Question / ### Answer` format — must be **identical** at training and inference time
5. **Saving**: Only adapter weights saved (`model.save_pretrained(ADAPTER_DIR)`), not the full model

For this project the target output is a JSON action list rather than SQL, but the QLoRA fine-tuning pipeline is identical.

## Running the Pipeline

```bash
# Execute trajectories against sales data and write results
python run_pipeline.py
# (default: reads sample_sales.csv + sample_trajectories.json → pipeline_results.txt)
# Update the paths in __main__ to use sales_data.csv + agent_trajectories_2k.json
```

## Prompt Design Notes

- The training data (`agent_trajectories_2k.json`) contains only `query` + `actions` (no computed `answer`). The answer must be obtained by executing the actions via `ToolExecutor`.
- The prompt must include the CSV schema so the model knows valid column names.
- Output must be parseable JSON — use a strict template that terminates cleanly after the closing `}`.
