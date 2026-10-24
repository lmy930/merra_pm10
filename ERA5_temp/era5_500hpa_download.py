"""ERA5 500 hPa 风场：旧 A 方法；三种模式；按自然月保存。
"""

import argparse
import calendar
import json
import shutil
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import cdsapi
import multiurl
import numpy as np
import xarray as xr

DATASET = "reanalysis-era5-pressure-levels"
AREA = [59, 50, 29, 138]                 # 北、西、南、东
LAT = np.arange(59, 28.999, -0.25)      # 121 个纬度，北→南
LON = np.arange(50, 138.001, 0.25)      # 353 个经度，西→东


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--date", type=date.fromisoformat, help="测试一天，如 2026-05-31")
    g.add_argument("--month", help="下载一个完整月，如 2026-05")
    g.add_argument("--start", type=date.fromisoformat, help="起始日，配合 --end")
    p.add_argument("--end", type=date.fromisoformat, help="结束日，包含当天")
    p.add_argument("--mode", choices=["links", "auto", "download"], default="auto")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "era5_500hpa_data")
    p.add_argument("--links-file", type=Path, help="download 模式导入已有的单个链接 TXT")
    p.add_argument("--refresh", action="store_true", help="链接失效时，links/auto 模式重新获取链接")
    p.add_argument("--keep-raw", action="store_true", help="保留下载原件，便于核对或测速")
    a = p.parse_args()

    if a.month:
        start = date.fromisoformat(a.month + "-01")
        end = start.replace(day=calendar.monthrange(start.year, start.month)[1])
    elif a.date:
        start = end = a.date
    else:
        start, end = a.start, a.end

    if bool(a.start) != bool(a.end):
        p.error("--start 和 --end 必须同时提供，且不能与 --date/--month 混用")
    if start < date(1940, 1, 1) or end < start:
        p.error("日期范围无效：起始日期须不早于 1940-01-01，结束日期须不早于起始日期")
    if a.mode == "download" and a.refresh:
        p.error("download 模式不提交请求，不能使用 --refresh")
    if a.links_file and a.mode != "download":
        p.error("--links-file 仅用于 download 模式")

    return a, start, end


def month_tasks(start, end):
    """自动按自然月拆分；单日测试只请求当天的 24 小时。"""
    while start <= end:
        last = start.replace(day=calendar.monthrange(start.year, start.month)[1])
        stop = min(last, end)
        full = start.day == 1 and stop == last
        token = start.strftime("%Y%m") if full else (
            f"{start:%Y%m%d}_test" if start == stop else f"{start:%Y%m%d}-{stop:%Y%m%d}_test")
        filename = f"ERA5_500hPa_uv_{token}_hourly.nc"

        request = {
            "product_type": ["reanalysis"],
            "variable": ["u_component_of_wind", "v_component_of_wind"],
            "pressure_level": ["500"],
            "year": [f"{start.year:04d}"],
            "month": [f"{start.month:02d}"],
            "day": [f"{d:02d}" for d in range(start.day, stop.day + 1)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": AREA,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }

        times = np.arange(
            np.datetime64(start, "h"),
            np.datetime64(stop + timedelta(days=1), "h"),
            np.timedelta64(1, "h"),
        ).astype("datetime64[ns]")

        yield token, filename, request, times
        start = stop + timedelta(days=1)


def read_data(path, times):
    """整理坐标并做必要检查，不插值、不平均、不改变浮点精度。"""
    source = path

    # 兼容 CDS 返回“包含一个 NetCDF 的 ZIP”的情况。
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith((".nc", ".nc4"))]
            if len(names) != 1:
                raise ValueError("ZIP 中不是单个 NetCDF，需检查后处理")
            source = path.with_name(path.name + ".unpacked.nc")
            with z.open(names[0]) as src, source.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    with xr.open_dataset(source, engine="netcdf4") as src:
        ds = src.rename({"time": "valid_time"}) if "time" in src.dims else src

        if "pressure_level" not in ds.coords and "level" in ds.coords:
            ds = ds.rename({"level": "pressure_level"})

        pressure = ds["pressure_level"]
        units = str(pressure.attrs.get("units", "")).lower().replace(" ", "")
        if (
            pressure.size != 1
            or float(pressure.values.reshape(-1)[0]) != 500
            or units not in (
                "", "hpa", "hectopascal", "hectopascal(hpa)",
                "mb", "mbar", "millibar", "millibars",
            )
        ):
            raise ValueError("数据不是预期的单一 500 hPa 气压层")

        if "pressure_level" in ds.dims:
            ds = ds.isel(pressure_level=0)

        if "expver" in ds:
            ds = ds.assign_coords(expver=ds.expver.astype(int))
        if "expver" in ds.dims:
            ds = ds.sel(expver=1)
        elif "expver" in ds and np.any(ds.expver.values != 1):
            raise ValueError("数据含 ERA5T，未作为正式 ERA5 保存")

        ds = (
            ds[["u", "v"]]
            .sortby("valid_time")
            .sortby("latitude", ascending=False)
            .sortby("longitude")
        )

        for name, expected in (
            ("valid_time", times), ("latitude", LAT), ("longitude", LON)
        ):
            if not np.array_equal(ds[name].values, expected):
                raise ValueError(f"{name} 与请求的逐小时时间或 0.25° 网格不一致")

        ds = ds.transpose("valid_time", "latitude", "longitude").load()

        for name in ("u", "v"):
            values = ds[name].values
            if ds[name].attrs.get("units") not in (
                "m s-1", "m s**-1", "m/s", "m s^-1", "m s**(-1)"
            ):
                raise ValueError(f"{name} 的风速单位不符")
            if np.isinf(values).any() or np.isnan(values).all(axis=(1, 2)).any():
                raise ValueError(f"{name} 存在无穷值或整小时全部缺测")

        extra = [
            c for c in ds.coords
            if c not in ("valid_time", "latitude", "longitude", "pressure_level")
        ]
        ds = ds.drop_vars(extra)
        ds.pressure_level.attrs.update(units="hPa", standard_name="air_pressure")
        ds.attrs.update(
            source_dataset=DATASET,
            source_method="CDS API NetCDF",
            time_zone="UTC",
            history=(
                "Coordinate naming/order only; no interpolation or averaging; "
                "lossless compression"
            ),
        )

    if source != path:
        source.unlink()

    return ds


def main():
    a, start, end = arguments()
    tasks = list(month_tasks(start, end))

    if a.links_file and len(tasks) != 1:
        raise ValueError("导入单个链接时，请只选择一天或一个月")

    folders = {
        name: a.out.resolve() / name
        for name in ("links", "raw", "data")
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    print(
        f"脚本：{Path(__file__).resolve()}\n"
        f"方式：旧 A 同步调用；模式：{a.mode}",
        flush=True,
    )

    for token, filename, request, times in tasks:
        record_path = folders["links"] / f"{token}.json"
        txt_path = record_path.with_suffix(".txt")
        raw = folders["raw"] / filename
        output = folders["data"] / filename

        print(f"\n{filename}：{len(times)} 小时", flush=True)

        if output.exists() and a.mode != "links":
            read_data(output, times).close()
            print("已有文件校验通过，跳过。", flush=True)
            continue

        record = (
            json.loads(record_path.read_text(encoding="utf-8"))
            if record_path.exists() else {}
        )
        if record and (
            record["request"] != request or record["dataset"] != DATASET
        ):
            raise ValueError("已有记录参数不同，请使用新的 --out 目录")

        previous_url = record.get("url")

        if a.links_file:
            urls = [
                line.strip()
                for line in a.links_file.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]
            if len(urls) != 1:
                raise ValueError("--links-file 必须包含且仅包含一个下载链接")
            record = {
                "dataset": DATASET,
                "request": request,
                "url": urls[0],
                "imported_utc": datetime.now(timezone.utc).isoformat(),
            }

        result = None

        # 第一部分：获取链接。
        # auto 获取后继续下载；links 保存后结束本月。
        if (
            a.mode != "download"
            and (a.mode == "links" or not raw.exists())
            and (a.refresh or not record.get("url"))
        ):
            print("提交 CDS 请求……", flush=True)
            result = cdsapi.Client().retrieve(DATASET, request)
            record = {
                "dataset": DATASET,
                "request": request,
                "url": result.location,
                "ready_utc": datetime.now(timezone.utc).isoformat(),
            }

        if record.get("url"):
            # 链接临时有效；保存生成时间，不编造固定到期日期。
            record_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            txt_path.write_text(record["url"] + "\n", encoding="utf-8")
            if previous_url != record["url"]:
                raw.with_name(raw.name + ".part").unlink(missing_ok=True)

        if a.mode == "links":
            print(f"链接已保存，不下载：{txt_path}", flush=True)
            continue

        if not raw.exists():
            if not record.get("url"):
                raise ValueError(
                    "没有对应的已保存链接，请先运行 links 或使用 --links-file"
                )

            part = raw.with_name(raw.name + ".part")

            if result is not None:
                # 第二部分：刚获取链接，直接用旧 A 的方式下载。
                result.download(str(part))
            else:
                # 第三部分：用已保存的链接下载，不重新提交 CDS 请求。
                multiurl.download(
                    record["url"],
                    str(part),
                    stream=True,
                    resume_transfers=True,
                    maximum_retries=3,
                )

            part.replace(raw)

        ds = read_data(raw, times)
        temp = output.with_name(output.name + ".part")
        ds.to_netcdf(
            temp,
            engine="netcdf4",
            format="NETCDF4",
            encoding={
                v: {"zlib": True, "complevel": 1}
                for v in ("u", "v")
            },
        )
        ds.close()
        temp.replace(output)

        if not a.keep_raw:
            raw.unlink()

        print(f"完成：{output}", flush=True)


if __name__ == "__main__":
    main()