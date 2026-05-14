from pathlib import Path
import shutil

ROOT_DIR = Path(__file__).parent

SOURCE_CSS = ROOT_DIR / "style.css"

NOTES_DIR = ROOT_DIR / "notes"

if not SOURCE_CSS.exists():
    print(f"找不到源文件: {SOURCE_CSS}")
    exit()

if not NOTES_DIR.exists():
    print(f"找不到 notes 文件夹: {NOTES_DIR}")
    exit()

print(" 开始同步 style.css...\n")

for folder in NOTES_DIR.rglob("*"):
    if folder.is_dir():
        target_css = folder / "style.css"

        try:
            shutil.copy2(SOURCE_CSS, target_css)
            print(f"已复制到: {target_css}")

        except Exception as e:
            print(f"失败: {target_css}")
            print(f"   错误: {e}")

print("\n全部完成！")