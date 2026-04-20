from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


# 计算项目根目录，后面读取 data 文件时会用到。
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"


def get_current_time(timezone_name: str = "Asia/Shanghai") -> str:
    """
    返回指定时区的当前时间，给 agent 作为可调用工具使用。

    注意事项：
    - timezone_name 需要使用标准 IANA 时区名称，例如 Asia/Shanghai。
    - 这个工具只负责“取时间”，不要把总结、解释之类的逻辑塞进工具里。
    - 工具返回值越单一越好，这样模型更容易稳定使用。
    """
    try:
        # ZoneInfo 使用标准 IANA 时区名称，比如 Asia/Shanghai。
        current_time = datetime.now(ZoneInfo(timezone_name))
    except Exception:
        return (
            f"Invalid timezone: {timezone_name}. "
            "Use a standard IANA timezone like Asia/Shanghai or America/Los_Angeles."
        )

    return current_time.strftime("%Y-%m-%d %H:%M:%S %Z")


def read_local_note(filename: str = "notes.txt") -> str:
    """
    读取 data 目录下的 txt 文件内容，避免 agent 直接访问任意路径。

    注意事项：
    - filename 最终只会保留文件名本身，不能跨目录读取其他文件。
    - 当前示例只支持 .txt，目的是先让你专注理解 Tool 调用流程。
    - 如果以后要支持更多格式，建议新建专门的工具，而不是把一个函数做得过于复杂。
    - 工具最好返回“可直接消费”的文本结果，避免把底层细节暴露给模型。
    """
    # 只保留文件名本身，防止传入类似 ../../secret.txt 这样的路径。
    safe_name = Path(filename).name
    note_path = DATA_DIR / safe_name

    # 这里只允许读取 txt，方便新手先聚焦在工具调用，而不是文件解析细节。
    if note_path.suffix.lower() != ".txt":
        return "Only .txt files are supported in this starter project."

    if not note_path.exists():
        # 如果文件不存在，就把当前可用的 txt 文件列出来，方便用户重试。
        available_files = sorted(path.name for path in DATA_DIR.glob("*.txt"))
        if not available_files:
            return "No note files were found in the data directory."

        return (
            f"File not found: {safe_name}. "
            f"Available files: {', '.join(available_files)}"
        )

    # 读取 UTF-8 文本内容并去掉首尾空白。
    return note_path.read_text(encoding="utf-8").strip()
