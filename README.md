# Blog-Learning

这是一个基于 [Hexo](https://hexo.io/) 的个人博客项目，使用 NexT 主题，站点部署在 GitHub Pages。

站点地址：<https://blueee0.github.io/my-blog>

## 环境要求

- Node.js 20.19.0 或更高版本
- npm

## 本地运行

首次克隆仓库后初始化主题子模块：

```bash
git submodule update --init --recursive
```

安装依赖并启动本地预览：

```bash
npm ci
npm run server
```

启动后访问 <http://localhost:4000/my-blog/>。

## 常用命令

```bash
npm run build    # 生成静态文件到 public/
npm run clean    # 清理生成文件和缓存
npm run server   # 启动本地开发服务器
```

## 写作

博客文章位于 `source/_posts/`。可以直接新建 Markdown 文件，也可以使用 Hexo CLI 创建文章：

```bash
npx hexo new post "文章标题"
```

## 部署

向 `main` 分支推送代码，或手动运行 `.github/workflows/pages.yml`，会触发 GitHub Pages 构建和部署流程。

工作流会通过 `submodules: recursive` 自动下载 NexT 主题子模块，确保云端构建环境包含 `themes/next/`。
