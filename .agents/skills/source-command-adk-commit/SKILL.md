---
name: "source-command-adk-commit"
description: "TTADK Standard Commit Command"
---

# source-command-adk-commit

Use this skill when the user asks to run the migrated source command `adk-commit`.

## Command Template

## Execution Rules

- Execute commands **strictly in order**
- **DO NOT run any extra commands** beyond this flow
**DO NOT run pre-checks or exploratory commands**, including:
   - `git diff --stat`
   - `git remote get-url`
- Pre-checks for submodule status are allowed and recommended
- Trust the defined steps and proceed directly

***

## Single Repository

### Steps

1. **Stage && Collect context**
   ```bash
      git add -A
      FILES=$(git diff --cached --name-only)
      DIFF=$(git diff --cached)

      if [ -z "$FILES" ]; then
         echo "No changes to commit" && exit 0
      fi
   ```
2. **Generate commit message**

   Model uses full $DIFF to generate conventional commit message

   Format:
   ```
   <type>(<scope>): <subject>
   ```
   Types: feat, fix, docs, style, refactor, test, chore

   Rules:
   - ≤72 chars
   - imperative mood
   - reflect actual intent (not file ops)
   - **no emoji**
3. **Commit & push**
   ```bash
      git commit -m "$MESSAGE"
      git push -u origin $(git branch --show-current)
   ```

***

## Multi-Repo / Submodules

1. **Check submodule status first**:
   ```bash
   git submodule foreach 'git status -s'
   ```
   This shows which submodules have changes. Only process those with non-empty output.

2. **Handle submodules with changes**: For each submodule with changes:
   - `cd <submodule>` and run Single Repository commit flow (stage/commit/push)
   - Skip submodules with no changes

3. **Commit main repo last** (includes submodule reference updates)

4. **Order matters**: Submodules first, then main repo — remote can't reference non-existent submodule commits

***

## Constraints

- DO NOT modify code
- On commit/push failure → **stop and report**
- Do NOT resolve conflicts automatically
- No remote → commit only
- No changes → skip

***

## Output

```
## [OK] Commit Summary

1. repo-a - [OK] feat: xxx -> Pushed | MR: <merge request URL>
2. repo-b - [SKIP] No changes
3. repo-c - [FAIL] Push failed. Reason: xxxx
```
