---
name: lark-cli
version: 1.0.0
description: "飞书 Lark CLI：登录认证、下载飞书文档（含图片/画板）、IM 消息收发、上传文件、读写文档。运行 lark-cli --help 探索所有可用命令。"
metadata:
  requires:
    bins: ["lark-cli"]
  cliHelp: "lark-cli --help"
---

# lark-cli

> 命令细节均可通过 `lark-cli <命令> --help` 探索。

## 安装 / 更新

```bash
# 安装 CLI + Skills
npm install -g @larksuite/cli
npx skills add larksuite/cli -y -g

# 更新（CLI 和 Skills 同时更新）
npm update -g @larksuite/cli && npx skills add larksuite/cli -g -y
```

> 更新后需**重启 AI Agent** 以加载最新 Skills。

## 认证

### 首次初始化

在**后台**执行下面的命令，命令阻塞并输出一个授权 URL——从输出中提取该 URL，立即用 `open` 打开浏览器，引导用户在浏览器完成应用创建，命令自动退出：

```bash
lark-cli config init --new
# 提取输出中的授权 URL 后执行：
open "<授权 URL>"
```

### 登录授权

初始化完成后，同样以**后台**方式发起 OAuth 授权，提取 URL 后 `open` 打开：

```bash
lark-cli auth login --recommend
# 提取输出中的授权 URL 后执行：
open "<授权 URL>"
```

> 多次 `auth login` 的 scope 会**累积**（增量授权），无需重新登录。

### 权限不足处理

错误响应中包含：
- `permission_violations`：缺失的 scope（N 选 1）
- `hint`：建议的修复命令

按 `hint` 补充授权，以**后台**方式执行并 `open` URL：

```bash
lark-cli auth login --scope "<missing_scope>"
```

### 验证

```bash
lark-cli auth status
```

## 只读块（readonly-block）说明

| 标签 | 处理方式 |
|------|---------|
| `<readonly-block type="isv">` | 根据上下文标题直接生成对应 Mermaid 代码块 |
| `<readonly-block type="iframe" href="...">` | 保留 href 中的原始 URL |

---

## 下载飞书文档（含图片、画板）

### 第 1 步：获取文档内容

```bash
# 获取全文（默认 XML 格式）
lark-cli docs +fetch --api-version v2 --doc "文档 URL 或 token" --as user

# 获取 Markdown 格式
lark-cli docs +fetch --api-version v2 --doc "<token>" --doc-format markdown --as user

# 先看目录结构，再按章节精读（推荐）
lark-cli docs +fetch --api-version v2 --doc "<token>" --scope outline --max-depth 3 --as user
lark-cli docs +fetch --api-version v2 --doc "<token>" --scope section --start-block-id <标题id> --detail with-ids --as user
```

### 第 2 步：下载图片 / 文件

文档内容里的素材以 XML 标签返回：

```xml
<img token="..." url="https://..."/>       <!-- 图片 -->
<source token="..." url="https://..."/>    <!-- 文件 -->
<whiteboard token="..."/>                  <!-- 画板 -->
```

- `<img>` / `<source>` 有 `url` 时，直接 HTTP GET 下载，无需调 CLI。
- 没有 `url` 时，用 token 下载：

```bash
lark-cli docs +media-download --token "<file_token>" --output ./asset --as user
```

### 第 3 步：下载画板缩略图

> **⚠️ 路径限制**：`--output` 只接受相对路径。需先 `cd` 到目标目录，再用 `.`。

```bash
mkdir -p <目标目录>
cd <目标目录>
lark-cli docs +media-download --type whiteboard --token "<whiteboard_token>" --output . --as user
```

---

## 创建 / 更新文档

```bash
# 创建文档（v2 API）
# 注意：v2 使用 --content 而非 --markdown，标题需写在内容的第一个 H1 里，无 --title 参数
lark-cli docs +create --api-version v2 --content "# 标题\n\n## 内容" --doc-format markdown --as user

# 从文件读取（@file 语法）
# ⚠️ 路径限制：@file 只接受相对路径，不支持绝对路径。需先 cd 到文件所在目录，再用 @./filename
cd <文件所在目录>
lark-cli docs +create --api-version v2 --content @./doc.md --doc-format markdown --as user

# 指定父目录
lark-cli docs +create --api-version v2 --content @./doc.md --doc-format markdown --parent-token fldcnXXXX --as user

# 放入个人空间
lark-cli docs +create --api-version v2 --content @./doc.md --doc-format markdown --parent-position my_library --as user
```

```bash
# 追加内容（v2 API）
lark-cli docs +update --api-version v2 --doc "<doc_id_or_url>" --command append --content "## 新章节\n\n内容" --doc-format markdown --as user

# 定位替换（范围）
lark-cli docs +update --api-version v2 --doc "<doc_id>" --command str_replace \
  --pattern "旧内容" --content "新内容" --doc-format markdown --as user
```

### 插入本地图片

`--content` 不支持本地图片路径，需在创建文档后单独调用 `+media-insert`：

```bash
# 追加到文档末尾
lark-cli docs +media-insert --doc "<doc_url_or_id>" --file ./image.png --as user

# 插入到指定文字附近（图片会插入到匹配块的顶层容器之后）
lark-cli docs +media-insert --doc "<doc_url_or_id>" --file ./image.png \
  --selection-with-ellipsis "目标文字" --as user

# 插入到指定文字之前
lark-cli docs +media-insert --doc "<doc_url_or_id>" --file ./image.png \
  --selection-with-ellipsis "目标文字" --before --as user

# 带对齐和标题
lark-cli docs +media-insert --doc "<doc_url_or_id>" --file ./image.png \
  --align center --caption "图片说明" --as user
```

**含图片的完整写入流程**：

1. 在 Markdown 中用占位文字（如 `[图：架构图]`）标记图片位置
2. `+create` 创建文档，记录返回的 `doc_id`
3. 对每张图片调用 `+media-insert --selection-with-ellipsis "[图：架构图]"`，图片插入到占位文字块后

### 读取白板 Mermaid 源码

`+media-download --type whiteboard` 只能下载缩略图（图片），要获取白板中的 Mermaid/PlantUML 源码，需使用 `whiteboard +query`：

```bash
# 导出为 Mermaid/代码格式（输出到终端）
lark-cli whiteboard +query --whiteboard-token "<whiteboard_token>" --output_as code --as user

# 导出到文件
lark-cli whiteboard +query --whiteboard-token "<whiteboard_token>" --output_as code --output ./diagram.mmd --as user

# 导出原始节点结构（JSON）
lark-cli whiteboard +query --whiteboard-token "<whiteboard_token>" --output_as raw --as user

# 导出为预览图片
lark-cli whiteboard +query --whiteboard-token "<whiteboard_token>" --output_as image --output ./diagram.png --as user
```

> whiteboard_token 通过 `lark-cli docs +fetch` 获取文档内容后，从返回的 `<whiteboard token="..."/>` 标签中提取。

### 插入 Mermaid / PlantUML 图表

Mermaid 通过飞书**白板块（whiteboard）**承载，需两步操作：

**第 1 步**：在文档中插入一个空白板块，获取其 `whiteboard_token`

```bash
# 先 fetch 文档，找到目标位置的 block-id
lark-cli docs +fetch --api-version v2 --doc "<doc_id>" --scope outline --as user

# 在指定块后插入白板块（block_insert_after），content 为飞书白板 XML 片段
lark-cli docs +update --api-version v2 --doc "<doc_id>" \
  --command block_insert_after --block-id "<target_block_id>" \
  --content '<whiteboard></whiteboard>' --as user
# 记录返回的 whiteboard block token
```

**第 2 步**：将 Mermaid 内容写入白板块

```bash
# 从字符串写入
lark-cli docs +whiteboard-update \
  --whiteboard-token "<whiteboard_token>" \
  --source "graph TD\n  A --> B" \
  --input_format mermaid --as user

# 从文件写入
lark-cli docs +whiteboard-update \
  --whiteboard-token "<whiteboard_token>" \
  --source @./diagram.mmd \
  --input_format mermaid --as user

# PlantUML 同理
lark-cli docs +whiteboard-update \
  --whiteboard-token "<whiteboard_token>" \
  --source @./diagram.puml \
  --input_format plantuml --as user

# 覆盖已有内容（默认 append）
lark-cli docs +whiteboard-update \
  --whiteboard-token "<whiteboard_token>" \
  --source @./diagram.mmd \
  --input_format mermaid --overwrite --as user
```

---

## IM 消息

### 查找群 chat-id

```bash
# 按群名搜索（返回 chat_id）
lark-cli im +chat-search --query "群名关键词" --as user
```

### 发送消息

```bash
# 发送文本到群
lark-cli im +messages-send --chat-id "oc_xxx" --text "Hello" --as bot

# 发送 Markdown 到群（自动转 post 格式）
lark-cli im +messages-send --chat-id "oc_xxx" --markdown "## 标题\n\n内容" --as bot

# 发送给个人（by open_id）
lark-cli im +messages-send --user-id "ou_xxx" --text "Hello" --as bot

# 发送图片 / 文件
lark-cli im +messages-send --chat-id "oc_xxx" --image ./image.png --as bot
lark-cli im +messages-send --chat-id "oc_xxx" --file ./report.pdf --as bot
```

### 回复消息

```bash
# 回复消息
lark-cli im +messages-reply --message-id "om_xxx" --text "回复内容" --as bot

# 回复到 thread（消息出现在 thread 流而不是主聊天）
lark-cli im +messages-reply --message-id "om_xxx" --text "回复内容" --reply-in-thread --as bot
```

### 读取消息

```bash
# 读取群聊记录
lark-cli im +chat-messages-list --chat-id "oc_xxx" --as user

# 搜索消息（关键词）
lark-cli im +messages-search --query "关键词" --as user

# 搜索消息（限定群 + 时间范围）
lark-cli im +messages-search --chat-id "oc_xxx" \
  --start "2026-05-01T00:00:00+08:00" \
  --end "2026-05-07T23:59:59+08:00" --as user
```

---

## 下载电子表格

```bash
lark-cli sheets +export --url "表格 URL" --file-extension xlsx --output-path "./report.xlsx" --as user
```

## 上传文件到云空间

```bash
lark-cli drive +upload --file ./report.pdf --folder-token fldbc_xxx --as user
lark-cli drive +upload --file ./report.pdf --name "季度总结.pdf" --as user
```
