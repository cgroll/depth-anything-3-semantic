# 📤 上传 Examples 到 Hugging Face Spaces 指南

## 🚨 问题：二进制文件被拒绝

Hugging Face Spaces 会拒绝大文件（>100MB）或二进制文件，需要使用 **Git LFS** 来上传。

## ✅ 解决方案

### 方案 1：使用 Git LFS（推荐）⭐

#### 步骤 1：配置 Git LFS

我已经为你创建了 `.gitattributes` 文件，配置了图片文件的 Git LFS：

```gitattributes
# Images in examples directory
workspace/gradio/examples/**/*.png filter=lfs diff=lfs merge=lfs -text
workspace/gradio/examples/**/*.jpg filter=lfs diff=lfs merge=lfs -text
workspace/gradio/examples/**/*.jpeg filter=lfs diff=lfs merge=lfs -text
workspace/gradio/examples/**/*.bmp filter=lfs diff=lfs merge=lfs -text
workspace/gradio/examples/**/*.tiff filter=lfs diff=lfs merge=lfs -text
workspace/gradio/examples/**/*.tif filter=lfs diff=lfs merge=lfs -text
```

#### 步骤 2：安装 Git LFS（如果还没有）

```bash
# macOS
brew install git-lfs

# Linux
sudo apt-get install git-lfs

# Windows
# 下载安装：https://git-lfs.github.com/
```

#### 步骤 3：初始化 Git LFS

```bash
cd /Users/bytedance/depth-anything-3

# 初始化 Git LFS
git lfs install

# 验证配置
git lfs track
```

#### 步骤 4：添加示例场景

```bash
# 创建 examples 目录
mkdir -p workspace/gradio/examples/my_scene

# 添加图像文件
cp your_images/* workspace/gradio/examples/my_scene/

# 添加文件到 Git LFS
git add workspace/gradio/examples/
git add .gitattributes

# 提交
git commit -m "Add example scenes with Git LFS"

# 推送到 Spaces
git push origin main
```

#### 步骤 5：验证

```bash
# 检查哪些文件使用了 LFS
git lfs ls-files

# 应该看到你的图片文件
```

---

### 方案 2：使用持久存储（推荐用于大量数据）⭐

如果示例场景很大，可以使用 Hugging Face Spaces 的持久存储功能。

#### 步骤 1：在 Spaces 设置中启用持久存储

1. 进入你的 Space 设置
2. 启用 "Persistent storage"
3. 设置存储大小（如 50GB）

#### 步骤 2：在应用启动时下载示例

修改 `app.py`，在启动时从外部源下载示例：

```python
import os
import subprocess

def download_examples():
    """Download examples from external source if not exists"""
    examples_dir = "workspace/gradio/examples"
    if not os.path.exists(examples_dir) or not os.listdir(examples_dir):
        print("Downloading example scenes...")
        # 从 Hugging Face Dataset 下载
        # 或从其他存储服务下载
        # subprocess.run(["huggingface-cli", "download", "dataset/examples", ...])
        pass

if __name__ == "__main__":
    download_examples()
    # ... 启动应用
```

#### 步骤 3：上传到 Hugging Face Dataset

```bash
# 安装依赖
pip install huggingface_hub datasets

# 上传到 Dataset
python -c "
from datasets import Dataset
from huggingface_hub import HfApi

# 创建 dataset 并上传
api = HfApi()
api.upload_folder(
    folder_path='workspace/gradio/examples',
    repo_id='your-username/your-examples-dataset',
    repo_type='dataset'
)
"
```

---

### 方案 3：压缩后上传（小文件）

如果图片文件较小（<100MB），可以压缩后上传：

```bash
# 压缩 examples 目录
tar -czf examples.tar.gz workspace/gradio/examples/

# 添加到 Git（作为普通文件）
git add examples.tar.gz
git commit -m "Add compressed examples"
git push

# 在应用启动时解压
# 在 app.py 中添加：
import tarfile
if not os.path.exists("workspace/gradio/examples"):
    print("Extracting examples...")
    tarfile.open("examples.tar.gz").extractall()
```

---

### 方案 4：运行时下载（推荐用于生产）⭐

在应用启动时从外部源下载示例场景：

#### 修改 `app.py`

```python
import os
import subprocess
from huggingface_hub import hf_hub_download

def setup_examples():
    """Setup examples directory by downloading if needed"""
    examples_dir = "workspace/gradio/examples"
    os.makedirs(examples_dir, exist_ok=True)
    
    # 如果 examples 目录为空，从外部源下载
    if not os.listdir(examples_dir):
        print("📥 Downloading example scenes...")
        
        # 方式 1: 从 Hugging Face Dataset 下载
        try:
            from datasets import load_dataset
            dataset = load_dataset("your-username/your-examples-dataset")
            # 处理并保存到 examples_dir
        except:
            pass
        
        # 方式 2: 从 URL 下载压缩包
        # import urllib.request
        # urllib.request.urlretrieve("https://...", "examples.zip")
        # 解压到 examples_dir
        
        print("✅ Examples downloaded")

if __name__ == "__main__":
    setup_examples()
    # ... 启动应用
```

---

## 🎯 推荐方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **Git LFS** | ✅ 简单直接<br>✅ 版本控制 | ⚠️ 需要 LFS 配额<br>⚠️ 大文件可能慢 | 小到中等示例（<1GB） |
| **持久存储** | ✅ 无大小限制<br>✅ 快速访问 | ⚠️ 需要手动上传<br>⚠️ 需要付费 | 大量示例（>1GB） |
| **运行时下载** | ✅ 不占用仓库空间<br>✅ 灵活 | ⚠️ 首次启动慢<br>⚠️ 需要网络 | 生产环境 |
| **压缩上传** | ✅ 简单 | ⚠️ 大小限制<br>⚠️ 需要解压 | 小文件（<100MB） |

---

## 📝 完整 Git LFS 设置步骤

### 1. 确保 Git LFS 已安装

```bash
git lfs version
# 如果未安装，按照上面的步骤安装
```

### 2. 初始化 Git LFS

```bash
cd /Users/bytedance/depth-anything-3
git lfs install
```

### 3. 检查 .gitattributes

确保 `.gitattributes` 包含图片文件配置（我已经添加了）。

### 4. 添加示例场景

```bash
# 创建场景
mkdir -p workspace/gradio/examples/scene1
cp your_images/* workspace/gradio/examples/scene1/

# 添加文件
git add workspace/gradio/examples/
git add .gitattributes

# 检查哪些文件会使用 LFS
git lfs ls-files

# 提交
git commit -m "Add example scenes with Git LFS"

# 推送
git push origin main
```

### 5. 验证上传

在 Spaces 中检查文件是否成功上传，图片文件应该显示为 LFS 指针。

---

## 🔧 故障排除

### 问题 1：Git LFS 配额不足

**解决方案：**
- 使用方案 2（持久存储）或方案 4（运行时下载）
- 压缩图片文件
- 只上传必要的示例

### 问题 2：推送失败

**检查：**
```bash
# 检查 LFS 文件
git lfs ls-files

# 检查 LFS 状态
git lfs status

# 重新推送
git push origin main --force
```

### 问题 3：文件仍然被拒绝

**可能原因：**
- `.gitattributes` 配置不正确
- 文件没有通过 LFS 添加

**解决：**
```bash
# 移除并重新添加
git rm --cached workspace/gradio/examples/**/*.png
git add workspace/gradio/examples/
git commit -m "Fix: Add images via Git LFS"
git push
```

---

## 💡 最佳实践

1. **小示例（<100MB）**：使用 Git LFS
2. **中等示例（100MB-1GB）**：使用 Git LFS 或持久存储
3. **大示例（>1GB）**：使用持久存储或运行时下载
4. **生产环境**：使用运行时下载，从外部源获取

---

## 📚 相关资源

- [Git LFS 文档](https://git-lfs.github.com/)
- [Hugging Face Spaces 文档](https://huggingface.co/docs/hub/spaces)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)

