# 📁 Examples 目录配置指南

## 📍 Examples 目录位置

### 默认位置

Examples 目录应该放在：

```
workspace/gradio/examples/
```

### 完整路径说明

根据 `app.py` 的配置：

```python
workspace_dir = os.environ.get("DA3_WORKSPACE_DIR", "workspace/gradio")
examples_dir = os.path.join(workspace_dir, "examples")
# 结果: workspace/gradio/examples/
```

## 📂 目录结构

Examples 目录应该按以下结构组织：

```
workspace/gradio/examples/
├── scene1/              # 场景 1
│   ├── 000.png          # 图像文件
│   ├── 010.png
│   ├── 020.png
│   └── ...
├── scene2/              # 场景 2
│   ├── 000.jpg
│   ├── 010.jpg
│   └── ...
└── scene3/              # 场景 3
    ├── image1.png
    ├── image2.png
    └── ...
```

### 要求

1. **每个场景一个文件夹**：每个场景应该有自己的文件夹
2. **文件夹名称**：文件夹名称会显示为场景名称
3. **图像文件**：支持 `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif` 格式
4. **第一张图像**：第一张图像（按文件名排序）会用作缩略图

## 🔧 配置方式

### 方式 1：使用默认路径（推荐）

直接创建目录：

```bash
mkdir -p workspace/gradio/examples
```

然后添加场景：

```bash
# 创建场景文件夹
mkdir -p workspace/gradio/examples/my_scene

# 复制图像文件
cp your_images/* workspace/gradio/examples/my_scene/
```

### 方式 2：使用环境变量

通过环境变量自定义位置：

```bash
# 设置环境变量
export DA3_WORKSPACE_DIR="/path/to/your/workspace"

# 然后 examples 会在 /path/to/your/workspace/examples
```

或在 `app.py` 中修改：

```python
workspace_dir = os.environ.get("DA3_WORKSPACE_DIR", "/custom/path/workspace")
```

### 方式 3：在 Hugging Face Spaces 中

在 Spaces 中，可以通过以下方式添加 examples：

1. **通过 Git 上传**：
   ```bash
   git add workspace/gradio/examples/
   git commit -m "Add example scenes"
   git push
   ```

2. **通过网页界面上传**：
   - 在 Spaces 的文件浏览器中创建 `workspace/gradio/examples/` 目录
   - 上传场景文件夹和图像

3. **使用持久存储**：
   - 如果使用持久存储，examples 会保存在持久存储中
   - 路径仍然是 `workspace/gradio/examples/`

## 📝 示例场景结构示例

### 示例 1：简单场景

```
workspace/gradio/examples/
└── indoor_room/
    ├── 000.png
    ├── 010.png
    ├── 020.png
    └── 030.png
```

### 示例 2：多个场景

```
workspace/gradio/examples/
├── outdoor_garden/
│   ├── frame_001.jpg
│   ├── frame_002.jpg
│   └── frame_003.jpg
├── office_space/
│   ├── img_000.png
│   ├── img_010.png
│   └── img_020.png
└── street_scene/
    ├── 000.png
    ├── 010.png
    └── 020.png
```

## 🔍 验证 Examples 目录

### 检查目录是否存在

```bash
# 检查默认位置
ls -la workspace/gradio/examples/

# 或使用 Python
python -c "
import os
workspace_dir = os.environ.get('DA3_WORKSPACE_DIR', 'workspace/gradio')
examples_dir = os.path.join(workspace_dir, 'examples')
print(f'Examples directory: {examples_dir}')
print(f'Exists: {os.path.exists(examples_dir)}')
if os.path.exists(examples_dir):
    scenes = [d for d in os.listdir(examples_dir) if os.path.isdir(os.path.join(examples_dir, d))]
    print(f'Found {len(scenes)} scenes: {scenes}')
"
```

### 检查场景信息

应用启动时会自动扫描 examples 目录，并在日志中显示：

```
Found 3 example scenes:
  - scene1 (5 images)
  - scene2 (10 images)
  - scene3 (8 images)
```

## 🚀 快速开始

### 1. 创建目录结构

```bash
# 在项目根目录
mkdir -p workspace/gradio/examples
```

### 2. 添加示例场景

```bash
# 创建场景文件夹
mkdir -p workspace/gradio/examples/my_first_scene

# 添加图像文件（复制你的图像）
cp /path/to/your/images/* workspace/gradio/examples/my_first_scene/
```

### 3. 验证

启动应用后，你应该能在 UI 中看到示例场景网格。

## 📊 在 Hugging Face Spaces 中

### 上传方式

1. **通过 Git**（推荐）：
   ```bash
   # 在本地准备 examples
   mkdir -p workspace/gradio/examples
   # ... 添加场景 ...
   
   # 提交并推送
   git add workspace/gradio/examples/
   git commit -m "Add example scenes"
   git push
   ```

2. **通过网页界面**：
   - 在 Spaces 的文件浏览器中
   - 创建 `workspace/gradio/examples/` 目录
   - 上传场景文件夹

### 注意事项

- **文件大小限制**：确保图像文件不超过 Spaces 的文件大小限制
- **持久存储**：如果使用持久存储，examples 会持久保存
- **缓存**：示例场景的结果会缓存在 `workspace/gradio/input_images/` 下

## 🔗 相关配置

### 环境变量

- `DA3_WORKSPACE_DIR`: 工作空间目录（默认：`workspace/gradio`）
- Examples 目录自动设置为：`{DA3_WORKSPACE_DIR}/examples`

### 代码中的配置

- `depth_anything_3/app/gradio_app.py`: `cache_examples()` 方法
- `depth_anything_3/app/modules/utils.py`: `get_scene_info()` 函数
- `depth_anything_3/app/modules/event_handlers.py`: `load_example_scene()` 方法

## ❓ 常见问题

### Q: Examples 目录不存在怎么办？

A: 应用会自动创建 `workspace/gradio/` 目录，但不会自动创建 `examples/` 子目录。你需要手动创建：

```bash
mkdir -p workspace/gradio/examples
```

### Q: 如何添加新的示例场景？

A: 只需在 `workspace/gradio/examples/` 下创建新文件夹并添加图像：

```bash
mkdir -p workspace/gradio/examples/new_scene
cp images/* workspace/gradio/examples/new_scene/
```

应用会在下次启动时自动检测新场景。

### Q: 场景名称如何显示？

A: 场景名称就是文件夹名称。例如：
- 文件夹：`workspace/gradio/examples/indoor_room/`
- 显示名称：`indoor_room`

### Q: 缩略图如何选择？

A: 缩略图是文件夹中按文件名排序后的第一张图像。

## 📝 总结

**Examples 目录位置：**
- **默认**：`workspace/gradio/examples/`
- **可通过环境变量**：`DA3_WORKSPACE_DIR` 自定义

**目录结构：**
```
workspace/gradio/examples/
├── scene1/
│   └── images...
├── scene2/
│   └── images...
└── scene3/
    └── images...
```

**快速创建：**
```bash
mkdir -p workspace/gradio/examples
# 然后添加场景文件夹和图像
```

