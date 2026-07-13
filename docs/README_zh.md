# xLLM 文档站

[English](./README.md) | [简体中文](./README.zh-CN.md)

本仓库是 [xLLM](https://github.com/xLLM-AI/xllm) 的 Astro + Starlight
文档站。xLLM 是面向国产 AI 加速器的高性能大语言模型推理框架。

站点基于 Starlight 和 `starlight-theme-rapide` 构建，包含自定义顶部导航、
中英文切换，以及用于将文档正文复制为 Markdown 的 `Copy page` 页面操作。

## 文档结构

文档内容按语言维护在两个并行目录中：

- 英文：`src/content/docs/en`
- 简体中文：`src/content/docs/zh`

当前站点路由规则如下：

- `/` 重定向到 `/en/`
- `/en/` 对应英文文档
- `/zh/` 对应简体中文文档

新增或移动页面时，请尽量在两个语言目录中保持相同的相对路径，方便 Starlight
在同一主题的不同语言版本之间切换。

## 项目结构

```text
.
├── astro.config.mjs          # Starlight、语言、侧边栏和组件配置
├── package.json              # npm 脚本和依赖
├── src/
│   ├── assets/               # 站点级资源，例如 logo
│   ├── components/           # Starlight 组件覆盖
│   ├── content/
│   │   └── docs/
│   │       ├── en/           # 英文文档
│   │       ├── zh/           # 简体中文文档
│   │       └── assets/       # 文档图片和架构图
│   ├── pages/index.astro     # 从 / 重定向到 /en/
│   └── styles/theme.css      # 项目主题样式
└── public/                   # 静态公共资源
```

## 本地开发

安装依赖：

```sh
npm install
```

启动本地开发服务：

```sh
npm run dev
```

构建生产站点：

```sh
npm run build
```

预览生产构建结果：

```sh
npm run preview
```

## 编辑文档

- 面向用户的文档内容放在 `src/content/docs/en` 和 `src/content/docs/zh`。
- 同一个页面同时存在中英文版本时，尽量保持两个文件的相对路径一致。
- 共享的文档图片放在 `src/content/docs/assets`。
- 新增需要出现在导航中的章节时，更新 `astro.config.mjs` 中的 `sidebar`
  配置。
- 提交前运行 `npm run build`，用于检查路由、frontmatter 和 Starlight
  内容问题。

## 相关仓库

- xLLM 源码：<https://github.com/xLLM-AI/xllm>
