# utree Command Reference

`utree` is the core CLI tool for this Skill, located at `$HOME/.local/bin/utree`.

All commands are executed in the **Skill root directory**.

---

## 1. context — Analyze Function Context

Obtain dependency chain, call relationships, and other context information for a specified function.

```bash
$HOME/.local/bin/utree context --file <file> --line <line> --output ${TMP_ROOT}/${file}_${line}.json
```

| Parameter | Description |
|------|------|
| `--file` | Source file relative path (relative to `PROJECT_ROOT`) |
| `--line` | Line number of the target function |
| `--output` | Output JSON file path |

> **Note**: The `utree context` command is NOT used for JS/TS scenarios — it is not applicable to JavaScript / TypeScript projects.

---

## 2. flush — Finalization (Post-execution, Required)

Completes data finalization and telemetry reporting. This is a **required pre-step** for Step 3 output reporting.

```bash
AGENT_SOURCE=<agent_name> MODEL_SOURCE=<model_name> TMP_ROOT=${TMP_ROOT} \
$HOME/.local/bin/utree flush --repo-path <root path of git repo>
```

| Environment Variable | Description |
|---------|------|
| `AGENT_SOURCE` | Name of the agent invoking the skill; select from: `trae`, `traecli`, `codex`, `claude code`, `aime`, `coze`, `unknown` |
| `MODEL_SOURCE` | Model name |
| `TMP_ROOT` | Temporary directory path created in Step 1. `utree flush` uses it to read runtime metadata such as `task_meta.json` |

| Parameter | Description |
|------|------|
| `--repo-path` | Absolute path to the git repository root directory |

---

## Error Handling

- If a `utree` command returns a non-zero exit code, read the stderr output and analyze the cause.
- When `context` command fails, degrade to manually Reading directly dependent source files.
- When `flush` command fails, still output the normal report to the user, but attach a finalization failure warning.
