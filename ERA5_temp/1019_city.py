"""多个年份匹配：每个省份一个目录，每个城市一个多年PM10 CSV。"""

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import posixpath
import re
import time
import warnings
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import netCDF4 as nc
import numpy as np


BASE = Path(r"D:\PythonProject\PythonProject\ERA5_temp")

PROVINCES = (
    "内蒙古自治区",
    "甘肃省",
    "宁夏回族自治区",
    "新疆维吾尔自治区",
)

VARIABLES = (
    "AIRDENS SO4 BCPHOBIC BCPHILIC OCPHOBIC OCPHILIC "
    "DU001 DU002 DU003 DU004 SS001 SS002 SS003 SS004"
).split()

# 沿用你提供的论文公式。
COEFFICIENTS = {
    v: (
        1.375 if v == "SO4"
        else 0.74 if v == "DU004"
        else 1.0
    )
    for v in VARIABLES
    if v != "AIRDENS"
}

FORMULA = (
    "("
    + " + ".join(
        f"{c:g}*{v}"
        for v, c in COEFFICIENTS.items()
    )
    + ")*AIRDENS*1e9"
)

UTC = timezone.utc
BEIJING = timezone(timedelta(hours=8))
WARNINGS = set()

NS = {
    "m": (
        "http://schemas.openxmlformats.org/"
        "spreadsheetml/2006/main"
    )
}

MAP_FIELDS = [
    "province", "city_id", "city", "region",
    "longitude", "latitude",
    "input_longitude", "input_latitude",
    "grid_longitude", "grid_latitude",
    "grid_id", "distance_km",
    "coordinate_format", "match_method",
]

FIELDS = MAP_FIELDS + [
    "time_utc", "time_beijing",
    "PM10", "PM10_unit",
    "model_level", "quality_flag",
]


def file_sha(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024**2), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()


def read_excel_rows(path):
    """读取实际非空单元格，跳过工作簿中的大量空白格式行。"""
    with ZipFile(path) as archive:
        strings = []

        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(
                archive.read("xl/sharedStrings.xml")
            )
            strings = [
                "".join(
                    t.text or ""
                    for t in item.findall(".//m:t", NS)
                )
                for item in root.findall("m:si", NS)
            ]

        workbook = ET.fromstring(
            archive.read("xl/workbook.xml")
        )
        sheet = workbook.find(
            "m:sheets/m:sheet", NS
        )

        if sheet is None:
            raise ValueError("Excel没有工作表")

        rid = sheet.get(
            "{http://schemas.openxmlformats.org/"
            "officeDocument/2006/relationships}id"
        )

        relationships = ET.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )

        targets = [
            r.get("Target")
            for r in relationships
            if (
                r.get("Id") == rid
                and r.get("TargetMode") != "External"
            )
        ]

        if len(targets) != 1:
            raise ValueError("无法定位Excel首个工作表")

        target = targets[0]
        sheet_path = (
            target.lstrip("/")
            if target.startswith("/")
            else posixpath.normpath("xl/" + target)
        )

        header = None
        result = []
        root = ET.fromstring(
            archive.read(sheet_path)
        )

        for row in root.findall(
            "m:sheetData/m:row", NS
        ):
            values = {}

            for cell in row.findall("m:c", NS):
                reference = cell.get("r", "")
                column = re.match(r"[A-Z]+", reference)

                if column is None:
                    continue

                if cell.find("m:f", NS) is not None:
                    raise ValueError(
                        f"{reference}含公式，"
                        "请把城市表转换为固定数值"
                    )

                value = cell.find("m:v", NS)
                text = (
                    value.text
                    if (
                        value is not None
                        and value.text is not None
                    )
                    else ""
                )

                if cell.get("t") == "s" and text:
                    text = strings[int(text)]

                elif cell.get("t") == "inlineStr":
                    text = "".join(
                        t.text or ""
                        for t in cell.findall(
                            "m:is//m:t", NS
                        )
                    )

                values[column[0]] = text.strip()

            if not any(values.values()):
                continue

            if header is None:
                header = values
                if (
                    len(set(header.values()))
                    != len(header)
                ):
                    raise ValueError("Excel列名重复")
            else:
                result.append(
                    {
                        name: values.get(column, "")
                        for column, name
                        in header.items()
                    }
                )

    required = {
        "id", "province", "name", "lat", "lon"
    }

    if (
        header is None
        or not required.issubset(header.values())
        or not result
    ):
        raise ValueError(
            "Excel必须包含id、province、name、lat、lon，"
            "并有数据"
        )

    return result


def coordinate(value, kind):
    value = float(value)

    if not math.isfinite(value):
        raise ValueError("经纬度必须为有限数值")

    if kind == "degrees_minutes":
        absolute = abs(value)
        degrees = math.floor(absolute)
        minutes = round(
            (absolute - degrees) * 100, 8
        )

        if minutes >= 60:
            raise ValueError(
                f"{value}不能解释为度.分"
            )

        value = math.copysign(
            degrees + minutes / 60, value
        )

    return value


def load_cities(path, coordinate_format):
    result = []
    ids = set()
    names = set()

    for row in read_excel_rows(path):
        province = row["province"]
        city = row["name"]
        city_id = row["id"]

        if province not in PROVINCES:
            raise ValueError(
                f"不属于四省区：{province}/{city}"
            )

        if (
            not city
            or not city_id
            or city_id in ids
            or (province, city) in names
        ):
            raise ValueError(
                "城市编号或省市名称为空/重复："
                f"{province}/{city}"
            )

        lon = coordinate(
            row["lon"], coordinate_format
        )
        lat = coordinate(
            row["lat"], coordinate_format
        )

        if (
            not -180 <= lon <= 180
            or not -90 <= lat <= 90
        ):
            raise ValueError(
                f"{city}经纬度越界，请检查列是否颠倒"
            )

        ids.add(city_id)
        names.add((province, city))

        result.append(
            dict(
                province=province,
                city_id=city_id,
                city=city,
                region=row.get("region", ""),
                longitude=lon,
                latitude=lat,
                input_longitude=float(row["lon"]),
                input_latitude=float(row["lat"]),
                coordinate_format=coordinate_format,
            )
        )

    return result


def nearest_grid(lon, lat):
    """按球面距离查找最近的MERRA-2原始网格中心。"""
    lons = -180 + np.arange(576) * 0.625
    lats = -90 + np.arange(361) * 0.5

    city_lon, city_lat = np.deg2rad([lon, lat])
    lat_rad = np.deg2rad(lats)

    a = (
        np.sin((lat_rad - city_lat) / 2) ** 2
    )[:, None] + (
        np.cos(lat_rad) * math.cos(city_lat)
    )[:, None] * (
        np.sin(
            (np.deg2rad(lons) - city_lon) / 2
        ) ** 2
    )

    flat = int(np.argmin(a))
    y, x = np.unravel_index(flat, a.shape)

    distance = (
        2 * 6371.0088
        * math.asin(
            math.sqrt(
                float(np.clip(a[y, x], 0, 1))
            )
        )
    )

    return (
        float(lons[x]),
        float(lats[y]),
        int(y * 576 + x),
        distance,
    )


def match_cities(cities):
    result = []

    for city in cities:
        lon, lat, grid_id, distance = nearest_grid(
            city["longitude"], city["latitude"]
        )

        result.append(
            dict(
                city,
                grid_longitude=lon,
                grid_latitude=lat,
                grid_id=grid_id,
                distance_km=distance,
                match_method=(
                    "spherical_nearest_native_center"
                ),
            )
        )

    return result


def read_array(variable, key):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        values = variable[key]

    for warning in caught:
        message = f"{variable.name}: {warning.message}"

        if message not in WARNINGS:
            WARNINGS.add(message)
            print(
                f"读取提示：{message}",
                flush=True,
            )

    return values


def read_times(ds):
    clock = ds["time"]
    calendar_name = getattr(
        clock, "calendar", "standard"
    )

    if calendar_name not in (
        "standard",
        "gregorian",
        "proleptic_gregorian",
    ):
        raise ValueError("时间日历不支持")

    dates = nc.num2date(
        read_array(clock, slice(None)),
        clock.units,
        calendar=calendar_name,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )

    times = [
        date.replace(tzinfo=UTC)
        for date in np.atleast_1d(dates)
    ]

    if not times:
        raise ValueError("日文件没有时次")

    return times


def check_source(ds):
    absent = set(VARIABLES) - set(ds.variables)

    if absent:
        raise ValueError(
            f"文件缺少变量：{sorted(absent)}；"
            "原始文件应包含14个变量"
        )

    if (
        "lev" not in ds.variables
        or not np.array_equal(
            np.asarray(
                read_array(
                    ds["lev"], slice(None)
                )
            ).reshape(-1),
            [72],
        )
    ):
        raise ValueError(
            "原始文件必须仅包含第72层"
        )

    for name in VARIABLES:
        expected = (
            "kg m-3"
            if name == "AIRDENS"
            else "kg kg-1"
        )

        if (
            getattr(ds[name], "units", "").strip()
            != expected
        ):
            raise ValueError(
                f"{name}单位不符，预期{expected}"
            )

        dims = ds[name].dimensions

        if (
            not {"time", "lat", "lon"}.issubset(dims)
            or set(dims)
            - {"time", "lev", "lat", "lon"}
        ):
            raise ValueError(
                f"{name}维度不符：{dims}"
            )


def read_axis(ds, name, step):
    units = getattr(
        ds[name], "units", ""
    ).lower()

    allowed = (
        {
            "degrees_east", "degree_east",
            "degrees_e", "degree_e",
        }
        if name == "lon"
        else {
            "degrees_north", "degree_north",
            "degrees_n", "degree_n",
        }
    )

    if units not in allowed:
        raise ValueError(
            f"{name}单位错误"
        )

    values = np.asarray(
        read_array(ds[name], slice(None)),
        dtype=float,
    )

    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            f"{name}不是有效一维坐标"
        )

    differences = np.diff(values)

    if len(values) > 1 and (
        not (
            np.all(differences > 0)
            or np.all(differences < 0)
        )
        or not np.allclose(
            np.abs(differences),
            step,
            rtol=0,
            atol=1e-6,
        )
    ):
        raise ValueError(
            f"{name}不是连续原始网格，未重采样"
        )

    if not np.allclose(
        values / step,
        np.rint(values / step),
        rtol=0,
        atol=1e-6,
    ):
        raise ValueError(
            f"{name}偏离原始网格中心"
        )

    return values


def grid_indices(lon, lat, matches):
    pairs = sorted(
        {
            (
                match["grid_latitude"],
                match["grid_longitude"],
            )
            for match in matches
        }
    )

    iy = []
    ix = []

    for grid_lat, grid_lon in pairs:
        y = int(
            np.argmin(abs(lat - grid_lat))
        )
        x = int(
            np.argmin(abs(lon - grid_lon))
        )

        if (
            abs(lat[y] - grid_lat) > 1e-6
            or abs(lon[x] - grid_lon) > 1e-6
        ):
            raise ValueError(
                "所需最近邻网格"
                f"({grid_lon}, {grid_lat})"
                "未包含在下载文件中；未改用较远网格"
            )

        iy.append(y)
        ix.append(x)

    lookup = {
        pair: index
        for index, pair in enumerate(pairs)
    }

    city_index = np.array(
        [
            lookup[
                (
                    match["grid_latitude"],
                    match["grid_longitude"],
                )
            ]
            for match in matches
        ]
    )

    return np.array(iy), np.array(ix), city_index


def selected_values(ds, name, index, iy, ix):
    source = ds[name]

    key = tuple(
        index
        if dim == "time"
        else 0
        if dim == "lev"
        else slice(None)
        for dim in source.dimensions
    )

    values = np.ma.asarray(
        read_array(source, key),
        dtype=np.float64,
    ).filled(np.nan)

    if [
        dim
        for dim in source.dimensions
        if dim not in ("time", "lev")
    ] == ["lon", "lat"]:
        values = values.T

    return values[iy, ix]


def selected_pm10(ds, index, iy, ix):
    mixed = np.zeros(len(iy))

    for name, coefficient in COEFFICIENTS.items():
        mixed += coefficient * selected_values(
            ds, name, index, iy, ix
        )

    density = selected_values(
        ds, "AIRDENS", index, iy, ix
    )

    if np.any(
        np.isfinite(density) & (density <= 0)
    ):
        raise ValueError(
            "匹配网格空气密度存在非正值，停止供核查"
        )

    return mixed * density * 1e9


def validate_times(times, year, allow_partial):
    if not times or any(
        instant.year != year
        or instant.hour % 3
        or instant.minute
        or instant.second
        or instant.microsecond
        for instant in times
    ):
        raise ValueError(
            f"{year}时间为空、年份不符"
            "或不在UTC三小时节点"
        )

    if (
        len(times) != len(set(times))
        or any(
            (second - first).total_seconds()
            != 10800
            for first, second in zip(
                times, times[1:]
            )
        )
    ):
        raise ValueError(
            f"{year}存在重复或缺失时次，"
            "请先核对输入文件"
        )

    full = (
        times[0]
        == datetime(year, 1, 1, tzinfo=UTC)
        and times[-1]
        == datetime(year, 12, 31, 21, tzinfo=UTC)
    )

    if not full and not allow_partial:
        raise ValueError(
            f"{year}仅覆盖"
            f"{times[0].isoformat()}至"
            f"{times[-1].isoformat()}；"
            "若有意测试这个连续片段，"
            "请加--allow-partial"
        )

    return full


def source_plan(data, year, source, allow_partial):
    annual = (
        data / "pm10" / f"{year}_pm10.nc4"
    )

    use_annual = (
        source == "annual"
        or (
            source == "auto"
            and annual.exists()
        )
    )

    raw = data / "raw" / str(year)

    paths = (
        [annual]
        if use_annual
        else (
            sorted(
                path
                for path in raw.iterdir()
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in (".nc", ".nc4")
                )
            )
            if raw.is_dir()
            else []
        )
    )

    if (
        not paths
        or any(
            not path.is_file()
            for path in paths
        )
    ):
        raise FileNotFoundError(
            f"{year}没有可用本地文件："
            f"{annual if use_annual else raw}"
        )

    ordered = []
    reference = None

    for path in paths:
        with nc.Dataset(path) as ds:
            if use_annual:
                if (
                    "PM10" not in ds.variables
                    or ds["PM10"].dimensions
                    != ("time", "lat", "lon")
                    or getattr(
                        ds["PM10"], "units", ""
                    ) != "ug m-3"
                    or "model_level" not in ds.variables
                    or int(
                        ds["model_level"].getValue()
                    ) != 72
                    or getattr(
                        ds, "formula", ""
                    ) != FORMULA
                ):
                    raise ValueError(
                        "年度PM10的变量、单位、"
                        f"层号或公式不符：{path}"
                    )
            else:
                check_source(ds)

            lon = read_axis(
                ds, "lon", 0.625
            )
            lat = read_axis(
                ds, "lat", 0.5
            )

            if reference is None:
                reference = lon, lat
            elif (
                not np.array_equal(
                    reference[0], lon
                )
                or not np.array_equal(
                    reference[1], lat
                )
            ):
                raise ValueError(
                    f"{year}文件网格不一致："
                    f"{path.name}"
                )

            ordered.append(
                (read_times(ds), path)
            )

    ordered.sort(
        key=lambda item: (
            item[0][0], str(item[1])
        )
    )

    times = [
        instant
        for values, _ in ordered
        for instant in values
    ]

    full = validate_times(
        times, year, allow_partial
    )

    manifest = [
        {
            "name": str(path.relative_to(data)),
            "sha256": file_sha(path),
        }
        for _, path in ordered
    ]

    fingerprint = sha256(
        json.dumps(
            manifest, sort_keys=True
        ).encode()
    ).hexdigest()

    return dict(
        kind="annual" if use_annual else "raw",
        ordered=ordered,
        lon=reference[0],
        lat=reference[1],
        times=times,
        full=full,
        inputs=manifest,
        fingerprint=fingerprint,
    )


@contextmanager
def output_lock(root):
    path = root / ".run.lock"

    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()

        stream.seek(0)

        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(
                    stream.fileno(),
                    msvcrt.LK_NBLCK,
                    1,
                )
            else:
                import fcntl

                fcntl.flock(
                    stream,
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )

        except OSError:
            raise RuntimeError(
                "另一个匹配进程正在写此目录，"
                "请等其结束"
            ) from None

        try:
            yield
        finally:
            stream.seek(0)

            if os.name == "nt":
                msvcrt.locking(
                    stream.fileno(),
                    msvcrt.LK_UNLCK,
                    1,
                )
            else:
                fcntl.flock(
                    stream, fcntl.LOCK_UN
                )


def publish(temporary, target):
    target.parent.mkdir(
        parents=True, exist_ok=True
    )

    for attempt in range(5):
        try:
            temporary.replace(target)
            return
        except PermissionError:
            if attempt == 4:
                raise PermissionError(
                    f"无法更新{target}，"
                    "请关闭Excel/WPS/预览后重运行；"
                    "临时文件保留"
                ) from None

            time.sleep(1)


def city_path(root, match):
    if re.search(
        r'[<>:"/\\|?*]', match["city"]
    ):
        raise ValueError(
            "城市名称不能作为Windows文件名："
            f"{match['city']}"
        )

    return (
        root / match["province"]
        / f"{match['city']}_PM10.csv"
    )


def csv_rows(path, match):
    if not path.exists():
        return

    previous_key = None
    previous = None

    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream)

        if reader.fieldnames != FIELDS:
            raise ValueError(
                f"CSV列结构不符，未覆盖：{path}"
            )

        for row in reader:
            if any(
                row[key] != str(match[key])
                for key in (
                    "province", "city_id", "city"
                )
            ):
                raise ValueError(
                    f"城市信息不符：{path}"
                )

            for key in (
                "longitude", "latitude",
                "grid_longitude", "grid_latitude",
            ):
                if (
                    not math.isfinite(float(row[key]))
                    or abs(
                        float(row[key]) - match[key]
                    ) > 1e-9
                ):
                    raise ValueError(
                        "城市/网格坐标发生变化，"
                        f"未混合保存：{path}"
                    )

            instant = datetime.fromisoformat(
                row["time_utc"]
            )

            if instant.tzinfo is None:
                raise ValueError(
                    f"CSV的UTC时间缺少时区：{path}"
                )

            instant = instant.astimezone(UTC)

            if (
                instant.hour % 3
                or instant.minute
                or instant.second
                or instant.microsecond
            ):
                raise ValueError(
                    f"CSV时间不在三小时节点：{path}"
                )

            if (
                previous_key is not None
                and instant < previous_key
            ):
                raise ValueError(
                    "旧CSV未按UTC时间排序，"
                    f"未覆盖：{path}"
                )

            if instant == previous_key:
                if row != previous:
                    raise ValueError(
                        "旧CSV有同一时次的冲突记录："
                        f"{path}"
                    )
                continue

            if previous is not None:
                yield previous_key, previous

            previous_key = instant
            previous = row

        if previous is not None:
            yield previous_key, previous


def merge_city(old, incoming, ready, match):
    """逐行归并：相同时次使用新值，其他已有时次保留。"""
    old_rows = iter(csv_rows(old, match))
    new_rows = iter(csv_rows(incoming, match))

    a = next(old_rows, None)
    b = next(new_rows, None)
    counts = Counter()

    with ready.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=FIELDS
        )
        writer.writeheader()

        while a is not None or b is not None:
            if (
                b is None
                or (
                    a is not None
                    and a[0] < b[0]
                )
            ):
                instant, row = a
                a = next(old_rows, None)
            else:
                instant, row = b

                if (
                    a is not None
                    and a[0] == b[0]
                ):
                    a = next(old_rows, None)

                b = next(new_rows, None)

            writer.writerow(row)
            counts[str(instant.year)] += 1

    return {
        "rows": sum(counts.values()),
        "rows_by_utc_year": dict(counts),
        "sha256": file_sha(ready),
    }


def process_year(plan, year, matches, root):
    stage = (
        root / "_work"
        / f"{year}_{time.time_ns()}"
    )
    stage.mkdir(parents=True)

    iy, ix, city_index = grid_indices(
        plan["lon"], plan["lat"], matches
    )

    streams = []
    writers = []
    missing = Counter()
    negative = Counter()

    try:
        for index, match in enumerate(matches):
            stream = (
                stage / f"{index}.csv"
            ).open(
                "w",
                encoding="utf-8-sig",
                newline="",
            )

            streams.append(stream)

            writer = csv.DictWriter(
                stream, fieldnames=FIELDS
            )
            writer.writeheader()
            writers.append(writer)

        for number, (
            times, path
        ) in enumerate(plan["ordered"], 1):
            with nc.Dataset(path) as ds:
                if read_times(ds) != times:
                    raise ValueError(
                        "读取期间源文件时间改变，"
                        "请停止下载后重试"
                    )

                names = (
                    ["PM10"]
                    if plan["kind"] == "annual"
                    else VARIABLES
                )

                for name in names:
                    if (
                        ds.data_model.startswith(
                            "NETCDF4"
                        )
                        and ds[name].chunking()
                        != "contiguous"
                    ):
                        ds[name].set_var_chunk_cache(
                            size=2 * 1024**2,
                            nelems=1009,
                            preemption=0.75,
                        )

                for index, instant in enumerate(times):
                    if plan["kind"] == "annual":
                        selected = selected_values(
                            ds, "PM10", index, iy, ix
                        )
                    else:
                        selected = selected_pm10(
                            ds, index, iy, ix
                        )

                    values = selected[city_index]

                    for match, value, writer in zip(
                        matches, values, writers
                    ):
                        key = match["city_id"]

                        missing[key] += int(
                            not np.isfinite(value)
                        )
                        negative[key] += int(
                            np.isfinite(value)
                            and value < 0
                        )

                        writer.writerow(
                            dict(
                                match,
                                time_utc=(
                                    instant.isoformat()
                                ),
                                time_beijing=(
                                    instant.astimezone(
                                        BEIJING
                                    ).isoformat()
                                ),
                                PM10=(
                                    float(value)
                                    if np.isfinite(value)
                                    else ""
                                ),
                                PM10_unit="ug/m3",
                                model_level=72,
                                quality_flag=(
                                    "missing_component_or_nonfinite"
                                    if not np.isfinite(value)
                                    else "negative_pm10"
                                    if value < 0
                                    else "ok"
                                ),
                            )
                        )

            if (
                number == 1
                or number % 10 == 0
                or number == len(plan["ordered"])
            ):
                print(
                    f"{year} 提取/计算 "
                    f"{number}/{len(plan['ordered'])}"
                    "个文件",
                    flush=True,
                )

    finally:
        for stream in streams:
            stream.close()

    outputs = {}

    for index, match in enumerate(matches):
        incoming = stage / f"{index}.csv"
        ready = stage / f"{index}.ready.csv"

        if (
            sum(
                1 for _ in csv_rows(incoming, match)
            )
            != len(plan["times"])
        ):
            raise ValueError(
                f"新年份行数不符：{match['city']}"
            )

        target = city_path(root, match)

        outputs[str(target.relative_to(root))] = (
            merge_city(
                target, incoming, ready, match
            )
        )

    # 所有城市文件准备好并关闭后，才更新正式文件。
    for index, match in enumerate(matches):
        publish(
            stage / f"{index}.ready.csv",
            city_path(root, match),
        )

    for path in stage.iterdir():
        path.unlink()

    stage.rmdir()

    return outputs, dict(missing), dict(negative)


def run_multi(
    cities_path,
    data,
    output,
    years,
    source="auto",
    allow_partial=False,
    skip_missing=False,
):
    started = time.perf_counter()

    matches = match_cities(
        load_cities(cities_path, "decimal")
    )

    config = {
        "excel_sha256": file_sha(cities_path),
        "formula": FORMULA,
        "columns": FIELDS,
        "coordinate_format": "decimal",
        "match_method": (
            "spherical_nearest_native_center"
        ),
    }

    signature = sha256(
        json.dumps(
            config, sort_keys=True
        ).encode()
    ).hexdigest()

    output.mkdir(
        parents=True, exist_ok=True
    )

    with output_lock(output):
        manifest_path = output / "metadata.json"

        state = (
            json.loads(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )
            if manifest_path.exists()
            else {}
        )

        if (
            state
            and state.get(
                "configuration_sha256"
            ) != signature
        ):
            raise ValueError(
                "城市表、公式或输出列已变化；"
                "请使用新的--output目录，"
                "避免混合不同配置"
            )

        verified = (
            bool(state.get("city_files"))
            or not state.get("processed_years")
        ) and all(
            (output / path).is_file()
            and file_sha(output / path)
            == record["sha256"]
            for path, record in state.get(
                "city_files", {}
            ).items()
        )

        if not verified:
            print(
                "现有文件与上次记录不一致，"
                "将重新核查请求年份；"
                "其他年份数据保留。",
                flush=True,
            )
            state["processed_years"] = {}

        state.update(
            configuration_sha256=signature,
            configuration=config,
            formula_reference=(
                "User-provided thesis screenshot, "
                "page 10, equation (1)"
            ),
            coordinate_crs=(
                "Not documented; original decimal "
                "values used without datum transformation"
            ),
        )

        mapping = (
            output / "city_grid_mapping.csv.tmp"
        )

        with mapping.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=MAP_FIELDS
            )
            writer.writeheader()
            writer.writerows(matches)

        publish(
            mapping,
            output / "city_grid_mapping.csv",
        )

        print(
            f"{len(matches)}个城市，"
            f"{len({m['grid_id'] for m in matches})}"
            "个独立网格；按城市合并多年数据。",
            flush=True,
        )

        skipped = []

        for year in sorted(set(years)):
            WARNINGS.clear()

            try:
                plan = source_plan(
                    data, year,
                    source, allow_partial,
                )
            except FileNotFoundError:
                if not skip_missing:
                    raise

                skipped.append(year)
                print(
                    f"{year}尚无本地数据，跳过。",
                    flush=True,
                )
                continue

            previous = state.get(
                "processed_years", {}
            ).get(str(year), {})

            if (
                previous.get("input_sha256")
                == plan["fingerprint"]
            ):
                print(
                    f"{year}：源文件和城市结果已校验，"
                    "跳过已完成年份。",
                    flush=True,
                )
                continue

            description = (
                "年度PM10，直接提取"
                if plan["kind"] == "annual"
                else "原始日文件，按公式计算"
            )

            print(
                f"{year}：使用{description}。",
                flush=True,
            )

            outputs, missing, negative = process_year(
                plan, year, matches, output
            )

            state["city_files"] = outputs

            state.setdefault(
                "processed_years", {}
            )[str(year)] = dict(
                input_sha256=plan["fingerprint"],
                source_type=plan["kind"],
                source_files=plan["inputs"],
                complete_source_year=plan["full"],
                time_count=len(plan["times"]),
                first_utc=(
                    plan["times"][0].isoformat()
                ),
                last_utc=(
                    plan["times"][-1].isoformat()
                ),
                missing_by_city_id=missing,
                negative_by_city_id=negative,
                read_warnings=sorted(WARNINGS),
            )

            state["updated_utc"] = (
                datetime.now(UTC).isoformat()
            )
            state["processing_seconds_this_run"] = (
                time.perf_counter() - started
            )

            temporary = manifest_path.with_suffix(
                ".json.tmp"
            )
            temporary.write_text(
                json.dumps(
                    state,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            publish(temporary, manifest_path)

            print(
                f"{year}完成，"
                "已更新每个城市的多年CSV。",
                flush=True,
            )

        state["last_run_requested_years"] = (
            sorted(set(years))
        )
        state["last_run_skipped_missing_years"] = (
            skipped
        )

        temporary = manifest_path.with_suffix(
            ".json.tmp"
        )
        temporary.write_text(
            json.dumps(
                state,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        publish(temporary, manifest_path)

        print(
            "处理结束，耗时"
            f"{time.perf_counter() - started:.1f}秒；"
            "缺少数据而跳过的年份："
            f"{skipped}",
            flush=True,
        )

    return output


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--cities",
        type=Path,
        default=BASE / "city四省.xlsx",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=BASE / "merra_data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            BASE / "merra_data/city_pm10_multi"
        ),
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--year",
        type=int,
        help="只处理一个年份",
    )
    group.add_argument(
        "--years",
        type=int,
        nargs="+",
        help="指定若干年份",
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=2015,
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
    )
    parser.add_argument(
        "--source",
        choices=("auto", "raw", "annual"),
        default="auto",
        help=(
            "auto优先使用年度PM10，"
            "缺少时使用原始日文件"
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="显式允许某年连续的部分时段",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help=(
            "没有本地数据的年份跳过，"
            "默认报错停止"
        ),
    )

    args = parser.parse_args()

    years = (
        [args.year]
        if args.year is not None
        else args.years
    )

    if years is None:
        if args.start_year > args.end_year:
            parser.error(
                "起始年不能大于结束年"
            )

        years = list(
            range(
                args.start_year,
                args.end_year + 1,
            )
        )

    if (
        not years
        or min(years) < 1980
        or max(years)
        > datetime.now(UTC).year
    ):
        parser.error(
            "年份必须在1980年至当前年份之间"
        )

    run_multi(
        args.cities,
        args.data,
        args.output,
        years,
        args.source,
        args.allow_partial,
        args.skip_missing,
    )


if __name__ == "__main__":
    main()


#& "D:\app\anaconda\envs\pytorch\python.exe" "D:\PythonProject\PythonProject\ERA5_temp\match_city_pm10_multi.py" --year 2024
#& "D:\app\anaconda\envs\pytorch\python.exe" "D:\PythonProject\PythonProject\ERA5_temp\1019_city.py" --start-year 2015 --end-year 2025