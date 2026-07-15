import os
from config import TEMP_MODEL_DIR, FINAL_MODEL_DIR


def select_model():
    """交互式选择模型文件，返回完整路径（test.py / gui.py 共用）"""
    m = input("Final model or temp model? [F/t]:").strip() or "f"
    if m.lower() == "f":
        all_entries = os.listdir(FINAL_MODEL_DIR)
        pth_files = sorted(
            [e for e in all_entries if os.path.isfile(os.path.join(FINAL_MODEL_DIR, e)) and e.lower().endswith(".pth")],
            reverse=True,
        )
        if not pth_files:
            raise FileNotFoundError(
                f"Directory {os.path.abspath(FINAL_MODEL_DIR)} does not contain any .pth file"
            )
        print(f"Found the following .pth files in the {FINAL_MODEL_DIR}:\n")
        for idx, filename in enumerate(pth_files, 1):
            print(f"\t{idx}. {filename}")
        print()
        num = int(input("Input number [1]: ") or "1")
        if num > len(pth_files):
            raise ValueError("Invalid number.")
        return os.path.join(FINAL_MODEL_DIR, pth_files[num - 1])
    elif m.lower() == "t":
        ch = int(input("Input checkpoint [100]: ").strip() or "100")
        return os.path.join(TEMP_MODEL_DIR, f"checkpoint_{ch}.pth")
    else:
        raise ValueError("Invalid option. Only f or t.")
