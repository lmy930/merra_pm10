"""用命令行选择国家，复用原脚本，不修改原文件。"""

import argparse
import importlib.util
from pathlib import Path

import sys


# 原脚本必须保留在同一个文件夹。
ORIGINAL_SCRIPT = (
    Path(__file__).resolve().parent / "1022_7_link.py"
)

AVAILABLE_COUNTRIES = [
    "蒙古国",
    "哈萨克斯坦",
    "吉尔吉斯斯坦",
    "塔吉克斯坦",
    "阿富汗",
    "巴基斯坦",
    "印度",
]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--countries",
        nargs="+",
        required=True,
        choices=AVAILABLE_COUNTRIES,
        help="本次只处理这些国家，多个国家用空格分开",
    )

    parser.add_argument(
        "--start-year",
        type=int,
        help="开始年份；不填则沿用原脚本",
    )

    parser.add_argument(
        "--end-year",
        type=int,
        help="结束年份；不填则沿用原脚本",
    )

    args = parser.parse_args()

    if not ORIGINAL_SCRIPT.is_file():
        raise FileNotFoundError(
            "找不到原脚本，请将两个文件放在同一文件夹："
            f"{ORIGINAL_SCRIPT}"
        )

    # 以模块方式加载，不执行原文件的程序入口。
    spec = importlib.util.spec_from_file_location(
        "neighbors_original_selected",
        ORIGINAL_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载原脚本。")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    required = (
        "COUNTRIES",
        "START_YEAR",
        "END_YEAR",
        "main",
    )

    if any(
        not hasattr(module, name)
        for name in required
    ):
        raise RuntimeError(
            "原脚本结构与预期不一致，"
            "尚未启动NASA任务。"
        )

    selected = list(
        dict.fromkeys(args.countries)
    )

    missing = [
        name
        for name in selected
        if name not in module.COUNTRIES
    ]

    if missing:
        raise ValueError(
            f"原脚本缺少这些国家的配置：{missing}"
        )

    # 只改变本次运行内存中的配置，不改写原文件。
    module.COUNTRIES = {
        name: module.COUNTRIES[name]
        for name in selected
    }

    if args.start_year is not None:
        module.START_YEAR = args.start_year

    if args.end_year is not None:
        module.END_YEAR = args.end_year

    if (
        not 1980 <= module.START_YEAR <= 9998
        or (
            module.END_YEAR is not None
            and not module.START_YEAR
            <= module.END_YEAR
            <= 9998
        )
    ):
        raise ValueError(
            "年份范围不正确，尚未启动NASA任务。"
        )

    print(
        "本次只处理：" + "、".join(selected),
        flush=True,
    )

    print(
        f"年份：{module.START_YEAR}—"
        f"{module.END_YEAR or 'NASA最新可用年份'}",
        flush=True,
    )

    print(
        "原脚本未修改；未选择的国家不进入处理流程。",
        flush=True,
    )

    module.main()


if __name__ == "__main__":
    main()

#& "D:\app\anaconda\envs\pytorch\python.exe" -u "D:\PythonProject\PythonProject\ERA5_temp\1020_select.py" --countries 阿富汗 巴基斯坦 印度 --start-year 2015
#& "D:\app\anaconda\envs\pytorch\python.exe" -u "D:\PythonProject\PythonProject\ERA5_temp\run_selected_neighbors.py" --countries 印度 --start-year 2015
#& "D:\app\anaconda\envs\pytorch\python.exe" -u "D:\PythonProject\PythonProject\ERA5_temp\run_selected_neighbors.py" --countries 印度 --start-year 2024 --end-year 2024
