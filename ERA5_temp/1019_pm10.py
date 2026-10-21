"""下载已有MERRA-2子集链接，逐日计算PM10，按年保存NetCDF4。"""

import argparse
import calendar
from datetime import datetime, timedelta, timezone
from getpass import getpass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import time
import warnings
import threading
from concurrent.futures import (
    ThreadPoolExecutor,
    wait,
    FIRST_COMPLETED,
    CancelledError,
)
from urllib.parse import urlsplit

import netCDF4 as nc
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from shapely import covers, points
from shapely.geometry import shape


# ===== 路径和数据配置 =====

DEFAULT_LINKS = (
    Path(r"D:\PythonProject\PythonProject\ERA5_temp\merra_links")
    if os.name == "nt"
    else Path(__file__).resolve().parent / "merra_links"
)

VARIABLES = (
    "AIRDENS SO4 BCPHOBIC BCPHILIC OCPHOBIC OCPHILIC "
    "DU001 DU002 DU003 DU004 SS001 SS002 SS003 SS004"
).split()

NAMES = {
    "宁夏": "Ningxia Ningxia Hui Autonomous Region",
    "新疆": "Xinjiang Uyghur Autonomous Region",
    "甘肃": "Gansu Province",
    "内蒙古": "Inner Mongolia Autonomous Region",
}

BOUNDARY_SHA = (
    "3a00467a0db9b4136facb5f2f3d0edbfd96adb15651cfdf63991da9281030e85"
)

BOUNDARY_URL = (
    "https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/"
    "5c25134028196d43ce97b5071934fd0cfc92f09f/"
    "releaseData/gbOpen/CHN/ADM1/geoBoundaries-CHN-ADM1.geojson"
)

EDL_HOSTS = {
    "urs.earthdata.nasa.gov",
    "harmony.earthdata.nasa.gov",
    "opendap.earthdata.nasa.gov",
    "data.gesdisc.earthdata.nasa.gov",
}

# 严格按你提供的论文截图：
# SO4系数1.375，DU004系数0.74，其余组分系数1。
# 不额外加入有机碳系数。
COEFFICIENTS = {
    variable: (
        1.375
        if variable == "SO4"
        else 0.74
        if variable == "DU004"
        else 1.0
    )
    for variable in VARIABLES
    if variable != "AIRDENS"
}

FORMULA = (
    "("
    + " + ".join(
        f"{coefficient:g}*{variable}"
        for variable, coefficient in COEFFICIENTS.items()
    )
    + ")*AIRDENS*1e9"
)

FILL = np.float32(nc.default_fillvals["f4"])
READ_WARNINGS = set()
NC_LOCK = threading.RLock()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_array(variable, key):
    """保留NetCDF自身的缺失值处理，记录读取警告。"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = variable[key]

    for warning in caught:
        message = f"{variable.name}: {warning.message}"
        if message not in READ_WARNINGS:
            READ_WARNINGS.add(message)
            print(
                f"读取提示（将写入报告）：{message}",
                flush=True,
            )

    return result


def read_year_links(path, year, now=None, check_expiry=True):
    now = now or datetime.now(timezone.utc)

    urls = list(
        dict.fromkeys(
            line.strip()
            for line in path.read_text(
                encoding="utf-8-sig"
            ).splitlines()
            if line.strip()
        )
    )

    if not urls:
        raise ValueError(f"{year}链接清单为空")

    dated = []

    for url in urls:
        parts = urlsplit(url)
        match = re.search(
            r"\.(\d{8})_subsetted\.nc4$",
            parts.path,
        )

        if (
            parts.scheme != "https"
            or parts.hostname != "harmony.earthdata.nasa.gov"
            or not parts.path.startswith("/service-results/")
            or not match
        ):
            raise ValueError(
                "清单不是已核对的Harmony子集结果链接，"
                "拒绝误下载全球源文件"
            )

        day = datetime.strptime(match[1], "%Y%m%d").date()

        if day.year != year:
            raise ValueError("链接清单包含其他年份")

        dated.append((day, url))

    days = [day for day, _ in dated]
    first = datetime(year, 1, 1).date()

    last = (
        max(days)
        if year == now.year
        else datetime(year, 12, 31).date()
    )

    expected = {
        first + timedelta(days=i)
        for i in range((last - first).days + 1)
    }

    if (
        len(set(days)) != len(days)
        or set(days) != expected
    ):
        raise ValueError(
            "年度链接日期有缺失、重复或超范围"
        )

    job_path = path.with_name(f"{year}_job.json")

    if job_path.exists():
        job = json.loads(
            job_path.read_text(encoding="utf-8")
        )

        if job.get("status") != "successful":
            raise ValueError(
                f"{year}任务记录尚未完整成功"
            )

        if job.get("job_id") and any(
            f"/{job['job_id']}/" not in urlsplit(url).path
            for url in urls
        ):
            raise ValueError(
                "年度TXT与job.json中的任务编号不一致"
            )

        expiration = job.get("data_expiration")

        if (
            check_expiry
            and expiration
            and datetime.fromisoformat(
                expiration.replace("Z", "+00:00")
            ) <= now
        ):
            raise ValueError(
                f"{year}NASA结果已过保存期；"
                "保留本地数据，先刷新或恢复该年度结果"
            )

    dated.sort(key=lambda item: item[0])

    return (
        [url for _, url in dated],
        datetime(
            last.year,
            last.month,
            last.day,
            21,
            tzinfo=timezone.utc,
        ),
    )


class EarthdataSession(requests.Session):
    """密码仅发送给明确列出的NASA HTTPS域名。"""

    def __init__(self, username, password):
        super().__init__()
        self.credentials = HTTPBasicAuth(username, password)
        self.auth = self.authorize
        self.headers["Accept-Encoding"] = "identity"

    def authorize(self, request):
        parsed = urlsplit(request.url)
        request.headers.pop("Authorization", None)

        if parsed.scheme != "https":
            raise ValueError("拒绝非HTTPS下载或重定向")

        if parsed.hostname in EDL_HOSTS:
            self.credentials(request)

        return request

    def rebuild_auth(self, prepared_request, response):
        self.authorize(prepared_request)


def check_netcdf(path):
    # netCDF/HDF5校验串行，网络下载仍然并发。
    with NC_LOCK, nc.Dataset(path) as ds:
        absent = set(VARIABLES) - set(ds.variables)

        if absent:
            raise ValueError(
                f"文件缺少变量：{sorted(absent)}"
            )

        if (
            "lev" not in ds.variables
            or not np.array_equal(
                np.asarray(ds["lev"][:]).reshape(-1),
                [72],
            )
        ):
            raise ValueError(
                "下载结果没有明确且唯一的lev=72坐标"
            )


def download(url, folder, session):
    """支持校验缓存及有服务器校验标识的断点续传。"""
    if urlsplit(url).scheme != "https":
        raise ValueError("链接必须是HTTPS")

    key = sha256(url.encode()).hexdigest()
    legacy = folder / f"{key[:24]}.nc4"

    day = re.search(
        r"\.(\d{8})_subsetted\.nc4$",
        urlsplit(url).path,
    )

    if legacy.exists() or legacy.with_suffix(".part").exists():
        target = legacy
    else:
        prefix = day[1] + "_" if day else ""
        target = folder / f"{prefix}{key[:24]}.nc4"

    receipt = target.with_suffix(".json")
    partial = target.with_suffix(".part")
    resume_file = target.with_suffix(".resume.json")

    if target.exists() and receipt.exists():
        record = json.loads(receipt.read_text())

        if (
            record.get("url_sha256") == key
            and record.get("sha256") == file_sha(target)
        ):
            check_netcdf(target)
            return target

    for attempt in range(3):
        offset = 0
        headers = {}

        resume = (
            json.loads(resume_file.read_text())
            if resume_file.exists()
            else {}
        )

        if partial.exists() and resume.get("validator"):
            offset = partial.stat().st_size

            if offset:
                headers = {
                    "Range": f"bytes={offset}-",
                    "If-Range": resume["validator"],
                }

        try:
            with session.get(
                url,
                headers=headers,
                stream=True,
                timeout=(30, 120),
            ) as response:
                if response.status_code == 416:
                    partial.unlink(missing_ok=True)
                    resume_file.unlink(missing_ok=True)
                    continue

                if response.status_code in (401, 403):
                    raise RuntimeError(
                        "下载授权被拒绝或链接过期；"
                        "请保留现有文件，先处理授权或刷新结果链接"
                    )

                response.raise_for_status()

                if "text/html" in response.headers.get(
                    "Content-Type", ""
                ).lower():
                    raise RuntimeError(
                        "服务器返回登录网页，未取得数据文件"
                    )

                if response.headers.get(
                    "Content-Encoding", "identity"
                ).lower() != "identity":
                    raise RuntimeError(
                        "服务器压缩了传输内容，"
                        "无法安全按字节续传；未拼接文件"
                    )

                expected_size = None

                if response.status_code == 206:
                    match = re.fullmatch(
                        r"bytes (\d+)-(\d+)/(\d+)",
                        response.headers.get(
                            "Content-Range", ""
                        ),
                    )

                    if not match or int(match[1]) != offset:
                        raise RuntimeError(
                            "续传范围不符，拒绝拼接文件"
                        )

                    expected_size = int(match[3])

                elif response.status_code == 200:
                    # 服务器忽略Range时，重新写入，不能追加。
                    offset = 0
                    length = response.headers.get(
                        "Content-Length"
                    )
                    expected_size = int(length) if length else None

                else:
                    raise RuntimeError(
                        f"非预期下载响应：{response.status_code}"
                    )

                validator = response.headers.get("ETag")

                if validator and validator.startswith("W/"):
                    validator = None

                validator = validator or response.headers.get(
                    "Last-Modified"
                )

                save(
                    resume_file,
                    {"validator": validator},
                )

                with partial.open(
                    "ab" if offset else "wb"
                ) as stream:
                    for chunk in response.iter_content(
                        1024 * 1024
                    ):
                        if chunk:
                            stream.write(chunk)

                if (
                    expected_size is not None
                    and partial.stat().st_size != expected_size
                ):
                    raise requests.ConnectionError(
                        "文件长度不完整"
                    )

            check_netcdf(partial)
            partial.replace(target)

            save(
                receipt,
                {
                    "url_sha256": key,
                    "sha256": file_sha(target),
                    "bytes": target.stat().st_size,
                },
            )

            resume_file.unlink(missing_ok=True)
            return target

        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ):
            if attempt == 2:
                raise RuntimeError(
                    "网络传输中断，"
                    "部分文件已保留供下次续传"
                ) from None

    raise RuntimeError(
        "下载没有完成，部分文件已保留"
    )


def load_provinces(path):
    if not path.exists():
        print(
            "缓存中没有省界，获取同一固定版本……",
            flush=True,
        )

        with requests.get(
            BOUNDARY_URL,
            timeout=(30, 120),
        ) as response:
            response.raise_for_status()
            raw = response.content

        if sha256(raw).hexdigest() != BOUNDARY_SHA:
            raise ValueError("下载的省界哈希不符")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    if file_sha(path) != BOUNDARY_SHA:
        raise ValueError(
            "省界文件哈希与固定版本不符"
        )

    features = json.loads(
        path.read_text(encoding="utf-8")
    )["features"]

    result = {}

    for province, name in NAMES.items():
        matches = [
            feature
            for feature in features
            if feature["properties"]["shapeName"] == name
        ]

        if len(matches) != 1:
            raise ValueError(
                f"{province}未唯一匹配"
            )

        geometry = shape(matches[0]["geometry"])

        if geometry.is_empty or not geometry.is_valid:
            raise ValueError(
                f"{province}边界无效"
            )

        result[province] = geometry

    return result


def grid_ids(lon, lat):
    return (
        np.rint((lat + 90) / 0.5).astype(np.int64) * 576
        + np.rint((lon + 180) / 0.625).astype(np.int64)
    )


def file_times(ds):
    variable = ds["time"]
    calendar_name = getattr(
        variable, "calendar", "standard"
    )

    if calendar_name not in (
        "standard",
        "gregorian",
        "proleptic_gregorian",
    ):
        raise ValueError("非预期时间日历")

    values = nc.num2date(
        read_array(variable, slice(None)),
        variable.units,
        calendar=calendar_name,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )

    result = [
        value.replace(tzinfo=timezone.utc)
        for value in np.atleast_1d(values)
    ]

    if not result or len(result) != len(set(result)):
        raise ValueError("文件时间为空或重复")

    return result


def axis_values(ds, name, step):
    coordinate_units = getattr(
        ds[name], "units", ""
    ).lower()

    expected_units = (
        {
            "degrees_east",
            "degree_east",
            "degrees_e",
            "degree_e",
        }
        if name == "lon"
        else {
            "degrees_north",
            "degree_north",
            "degrees_n",
            "degree_n",
        }
    )

    if coordinate_units not in expected_units:
        raise ValueError(
            f"{name}单位不是预期经纬度单位："
            f"{coordinate_units!r}"
        )

    values = np.asarray(
        read_array(ds[name], slice(None)),
        dtype=np.float64,
    )

    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            f"{name}不是有效一维坐标"
        )

    if len(values) > 1 and not (
        np.all(np.diff(values) > 0)
        or np.all(np.diff(values) < 0)
    ):
        raise ValueError(
            f"{name}坐标不单调或有重复"
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

    if len(values) > 1 and not np.allclose(
        np.abs(np.diff(values)),
        step,
        rtol=0,
        atol=1e-6,
    ):
        raise ValueError(
            f"{name}分辨率与原始网格不符"
        )

    return values


def available_cpu_count():
    counts = []

    for name in ("process_cpu_count", "cpu_count"):
        function = getattr(os, name, None)

        if function is not None:
            try:
                value = function()
                if value:
                    counts.append(value)
            except (OSError, NotImplementedError):
                pass

    if hasattr(os, "sched_getaffinity"):
        try:
            counts.append(
                len(os.sched_getaffinity(0))
            )
        except (OSError, NotImplementedError):
            pass

    return max(1, min(counts)) if counts else 1


def automatic_workers(cores):
    # 保守起点；网络并发不与CPU核数线性对应。
    return (
        1
        if cores <= 1
        else min(4, max(2, cores // 2))
    )


def province_mask(lon, lat, provinces):
    xx, yy = np.meshgrid(lon, lat)
    mask = np.zeros(xx.shape, dtype=np.uint8)

    for index, geometry in enumerate(
        provinces.values()
    ):
        mask[
            covers(geometry, points(xx, yy))
        ] |= 1 << index

    return mask


def read_slice(ds, name, index):
    source = ds[name]

    expected = (
        "kg m-3"
        if name == "AIRDENS"
        else "kg kg-1"
    )

    if getattr(source, "units", "").strip() != expected:
        raise ValueError(
            f"{name}单位不符，要求{expected}"
        )

    dims = source.dimensions

    if (
        not {"time", "lat", "lon"}.issubset(dims)
        or set(dims) - {"time", "lat", "lon", "lev"}
    ):
        raise ValueError(
            f"{name}维度不符：{dims}"
        )

    key = tuple(
        index
        if dim == "time"
        else 0
        if dim == "lev"
        else slice(None)
        for dim in dims
    )

    values = np.ma.asarray(
        read_array(source, key),
        dtype=np.float64,
    )

    if [
        dim for dim in dims
        if dim not in ("time", "lev")
    ] == ["lon", "lat"]:
        values = values.T

    return values.filled(np.nan)


def pm10_slice(ds, index, mask):
    mixed = np.zeros(
        mask.shape,
        dtype=np.float64,
    )

    for name, coefficient in COEFFICIENTS.items():
        mixed += coefficient * read_slice(
            ds, name, index
        )

    density = read_slice(
        ds, "AIRDENS", index
    )

    if np.any(
        np.isfinite(density[mask])
        & (density[mask] <= 0)
    ):
        raise ValueError(
            "省内AIRDENS存在非正值，"
            "未静默替换，请检查原始数据"
        )

    result = mixed * density * 1e9

    result[
        ~mask | ~np.isfinite(result)
    ] = FILL

    if np.any(
        np.abs(result[result != FILL])
        > np.finfo(np.float32).max
    ):
        raise ValueError(
            "PM10超过float32范围"
        )

    return result.astype(np.float32)


def validate_output(
    path, year, expected_stop, fingerprint
):
    digest = sha256()

    with nc.Dataset(path) as ds:
        if (
            ds.data_model != "NETCDF4"
            or ds.formula != FORMULA
            or ds.input_fingerprint != fingerprint
        ):
            raise ValueError(
                "已有输出的格式、公式或输入标识不符，"
                "未覆盖"
            )

        times = file_times(ds)
        first = datetime(
            year, 1, 1,
            tzinfo=timezone.utc,
        )

        if (
            times[0] != first
            or times[-1] != expected_stop
            or any(
                (second - first_time).total_seconds()
                != 10800
                for first_time, second in zip(
                    times, times[1:]
                )
            )
        ):
            raise ValueError(
                "年度文件时间覆盖不完整或存在缺口"
            )

        if (
            ds["PM10"].units != "ug m-3"
            or int(
                ds["model_level"].getValue()
            ) != 72
        ):
            raise ValueError(
                "年度文件单位或模式层不符"
            )

        for index in range(len(times)):
            values = (
                np.ma.asarray(
                    ds["PM10"][index]
                )
                .filled(FILL)
                .astype("<f4")
            )

            digest.update(values.tobytes())

        if digest.hexdigest() != ds.data_sha256:
            raise ValueError(
                "年度PM10回读数值校验失败"
            )

    return True


def promote(temporary, target):
    """所有NetCDF句柄关闭后，再进行改名。"""
    for attempt in range(5):
        if target.exists():
            raise FileExistsError(
                f"结果已经存在，未覆盖：{target}"
            )

        try:
            temporary.rename(target)
            return

        except PermissionError:
            if attempt == 4:
                raise PermissionError(
                    f"文件被占用，完整临时结果保留在"
                    f"{temporary}；关闭预览或打开文件的"
                    "软件后重运行"
                ) from None

            time.sleep(1)


def convert_year(
    files,
    year,
    output,
    provinces,
    expected_stop,
    fingerprint,
):
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    target = output / f"{year}_pm10.nc4"
    temporary = output / f"{year}_pm10.nc4.part"

    if target.exists():
        validate_output(
            target,
            year,
            expected_stop,
            fingerprint,
        )
        print(
            f"{year}：已有年度PM10通过回读校验，跳过。",
            flush=True,
        )
        return target

    if temporary.exists():
        try:
            validate_output(
                temporary,
                year,
                expected_stop,
                fingerprint,
            )

        except (
            OSError,
            AttributeError,
            KeyError,
            ValueError,
            IndexError,
        ):
            backup = output / (
                f"{year}_pm10.incomplete."
                f"{time.time_ns()}.part"
            )
            temporary.rename(backup)

            print(
                f"未完成的临时结果保留为{backup.name}；"
                "从本地日文件重新计算。",
                flush=True,
            )

        else:
            promote(temporary, target)
            print(
                f"{year}：已恢复完整临时文件，"
                "无需重新计算。",
                flush=True,
            )
            return target

    ordered = []

    for path in files:
        check_netcdf(path)

        with nc.Dataset(path) as ds:
            ordered.append(
                (min(file_times(ds)), path)
            )

    ordered.sort(
        key=lambda item: (
            item[0], str(item[1])
        )
    )

    if not ordered:
        raise ValueError(
            "没有可计算的原始文件"
        )

    started = time.perf_counter()

    with nc.Dataset(ordered[0][1]) as ds:
        lon = axis_values(ds, "lon", 0.625)
        lat = axis_values(ds, "lat", 0.5)

    mask = province_mask(
        lon, lat, provinces
    )

    world_lon = (
        -180 + np.arange(576) * 0.625
    )
    world_lat = (
        -90 + np.arange(361) * 0.5
    )

    world_mask = province_mask(
        world_lon, world_lat, provinces
    )

    wx, wy = np.meshgrid(
        world_lon, world_lat
    )
    xx, yy = np.meshgrid(lon, lat)

    actual_ids = set(
        grid_ids(
            xx[mask > 0],
            yy[mask > 0],
        )
    )

    expected_ids = set(
        grid_ids(
            wx[world_mask > 0],
            wy[world_mask > 0],
        )
    )

    if actual_ids != expected_ids:
        raise ValueError(
            "下载结果未覆盖四省全部目标网格中心"
        )

    rows, cols = np.where(mask > 0)

    ys = slice(
        rows.min(), rows.max() + 1
    )
    xs = slice(
        cols.min(), cols.max() + 1
    )

    cropped_mask = mask[ys, xs]
    out_lon = lon[xs]
    out_lat = lat[ys]

    digest = sha256()
    count = 0
    missing = 0
    negative = 0
    minimum = None
    maximum = None

    first = datetime(
        year, 1, 1,
        tzinfo=timezone.utc,
    )

    with nc.Dataset(
        temporary,
        "w",
        format="NETCDF4",
    ) as out:
        for name, size in (
            ("time", None),
            ("lat", len(out_lat)),
            ("lon", len(out_lon)),
        ):
            out.createDimension(name, size)

        out.Conventions = "CF-1.8"
        out.title = (
            "MERRA-2 model-level-72 PM10 "
            "over four Chinese provinces"
        )
        out.source_product = "M2I3NVAER 5.12.4"
        out.formula = FORMULA

        out.formula_reference = (
            "User-provided Peking University master's "
            "thesis screenshot, page 10, equation (1); "
            "full citation not supplied"
        )

        out.input_fingerprint = fingerprint
        out.boundary_rule = (
            "Native grid center covered by province "
            "polygon; outside union is missing"
        )
        out.boundary_sha256 = BOUNDARY_SHA

        out.province_bits = json.dumps(
            {
                name: 1 << index
                for index, name in enumerate(
                    provinces
                )
            },
            ensure_ascii=False,
        )

        out.created_utc = datetime.now(
            timezone.utc
        ).isoformat()

        out.numeric_method = (
            "float64 arithmetic, float32 storage; "
            "zlib lossless compression; no quantization"
        )

        for name, values, units, standard in (
            (
                "lat",
                out_lat,
                "degrees_north",
                "latitude",
            ),
            (
                "lon",
                out_lon,
                "degrees_east",
                "longitude",
            ),
        ):
            coordinate = out.createVariable(
                name, "f8", (name,)
            )
            coordinate[:] = values
            coordinate.units = units
            coordinate.standard_name = standard

        clock = out.createVariable(
            "time", "f8", ("time",)
        )
        clock.units = (
            f"hours since {year}-01-01 00:00:00"
        )
        clock.calendar = "standard"
        clock.standard_name = "time"
        clock.time_zone = "UTC"

        level = out.createVariable(
            "model_level", "i4"
        )
        level.assignValue(72)
        level.long_name = (
            "MERRA-2 native model level, "
            "72 is nearest to surface"
        )

        region = out.createVariable(
            "province_mask",
            "u1",
            ("lat", "lon"),
            zlib=True,
            complevel=2,
        )
        region[:] = cropped_mask
        region.long_name = (
            "Province membership bit mask; "
            "0 outside provinces"
        )

        pm = out.createVariable(
            "PM10",
            "f4",
            ("time", "lat", "lon"),
            fill_value=FILL,
            zlib=True,
            complevel=2,
            shuffle=True,
            fletcher32=True,
            chunksizes=(
                8,
                min(32, len(out_lat)),
                min(64, len(out_lon)),
            ),
        )

        pm.units = "ug m-3"
        pm.long_name = (
            "PM10 mass concentration reconstructed "
            "using the recorded formula"
        )
        pm.coordinates = "model_level"

        for number, (_, path) in enumerate(
            ordered, 1
        ):
            with nc.Dataset(path) as ds:
                if (
                    not np.array_equal(
                        axis_values(
                            ds, "lon", 0.625
                        ),
                        lon,
                    )
                    or not np.array_equal(
                        axis_values(
                            ds, "lat", 0.5
                        ),
                        lat,
                    )
                ):
                    raise ValueError(
                        "日文件网格不一致，未重采样"
                    )

                for name in VARIABLES:
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

                for index, instant in enumerate(
                    file_times(ds)
                ):
                    expected_time = (
                        first
                        + timedelta(hours=3 * count)
                    )

                    if (
                        instant != expected_time
                        or instant > expected_stop
                    ):
                        raise ValueError(
                            "时间重复、缺失、乱序"
                            "或超出该年清单范围"
                        )

                    values = pm10_slice(
                        ds,
                        index,
                        mask > 0,
                    )[ys, xs]

                    inside = values[
                        cropped_mask > 0
                    ]
                    finite = inside[
                        inside != FILL
                    ]

                    missing += int(
                        (inside == FILL).sum()
                    )
                    negative += int(
                        (finite < 0).sum()
                    )

                    if finite.size:
                        minimum = (
                            float(
                                min(
                                    minimum,
                                    finite.min(),
                                )
                            )
                            if minimum is not None
                            else float(finite.min())
                        )
                        maximum = (
                            float(
                                max(
                                    maximum,
                                    finite.max(),
                                )
                            )
                            if maximum is not None
                            else float(finite.max())
                        )

                    clock[count] = 3 * count
                    pm[count] = values

                    digest.update(
                        values.astype(
                            "<f4"
                        ).tobytes()
                    )

                    count += 1

            if (
                number == 1
                or number % 10 == 0
                or number == len(ordered)
            ):
                print(
                    f"{year} 计算/写入 "
                    f"{number}/{len(ordered)}个日文件，"
                    f"{count}个时次",
                    flush=True,
                )

        actual_stop = (
            first
            + timedelta(
                hours=3 * (count - 1)
            )
        )

        if actual_stop != expected_stop:
            raise ValueError(
                "实际文件没有覆盖链接清单结束时次"
            )

        out.data_sha256 = digest.hexdigest()
        out.time_count = count
        out.grid_count = int(
            (mask > 0).sum()
        )
        out.complete_calendar_year = int(
            count
            == (
                366
                if calendar.isleap(year)
                else 365
            ) * 8
        )

        out.source_manifest = json.dumps(
            [
                {
                    "name": path.name,
                    "sha256": file_sha(path),
                }
                for _, path in ordered
            ]
        )

        out.read_warnings = json.dumps(
            sorted(READ_WARNINGS),
            ensure_ascii=False,
        )

        out.missing_inside = missing
        out.negative_inside = negative
        out.pm10_min = (
            minimum
            if minimum is not None
            else np.nan
        )
        out.pm10_max = (
            maximum
            if maximum is not None
            else np.nan
        )
        out.processing_seconds = (
            time.perf_counter() - started
        )

    # 写入句柄已经关闭。重新读取所有时次校验。
    validate_output(
        temporary,
        year,
        expected_stop,
        fingerprint,
    )

    promote(temporary, target)

    print(
        f"完成：{target}，"
        f"{target.stat().st_size / 1024**2:.2f} MiB；"
        f"{count}时次",
        flush=True,
    )

    return target


def cached_file(url, folder):
    key = sha256(url.encode()).hexdigest()

    day = re.search(
        r"\.(\d{8})_subsetted\.nc4$",
        urlsplit(url).path,
    )[1]

    candidates = (
        folder / f"{key[:24]}.nc4",
        folder / f"{day}_{key[:24]}.nc4",
    )

    for path in candidates:
        if not path.is_file():
            continue

        receipt = path.with_suffix(".json")

        if receipt.exists():
            record = json.loads(
                receipt.read_text()
            )

            if (
                record.get("url_sha256") != key
                or record.get("sha256")
                != file_sha(path)
            ):
                raise ValueError(
                    f"已有缓存校验失败，未覆盖：{path}"
                )

        check_netcdf(path)

        with NC_LOCK, nc.Dataset(path) as ds:
            times = file_times(ds)

        first = datetime.strptime(
            day, "%Y%m%d"
        ).replace(tzinfo=timezone.utc)

        expected = [
            first + timedelta(hours=3 * index)
            for index in range(8)
        ]

        if times != expected:
            raise ValueError(
                "原始日文件时次不完整或与链接不符："
                f"{path}"
            )

        return path

    return None


def fetch_missing(
    urls,
    raw,
    existing,
    workers,
    credentials=None,
):
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    missing = [
        index
        for index, path in enumerate(existing)
        if path is None
    ]

    if not missing:
        return existing

    username, password = credentials or (
        input("Earthdata用户名：").strip(),
        getpass("密码（不显示、不保存）："),
    )

    sessions = []
    local = threading.local()
    stopped = threading.Event()

    def new_session(cookies=None):
        session = EarthdataSession(
            username, password
        )

        retry = Retry(
            total=3,
            read=0,
            backoff_factor=0.5,
            status_forcelist=(
                429, 500, 502, 503, 504
            ),
            allowed_methods=frozenset(
                ("GET", "HEAD")
            ),
            respect_retry_after_header=True,
        )

        session.mount(
            "https://",
            HTTPAdapter(max_retries=retry),
        )

        if cookies is not None:
            session.cookies.update(cookies)

        sessions.append(session)
        return session

    base = new_session()
    pool = None
    pending = {}

    completed = len(urls) - len(missing)
    started = time.perf_counter()

    try:
        print(
            "先完成首个缺失文件的认证，"
            "随后并发下载……",
            flush=True,
        )

        index = missing[0]
        existing[index] = download(
            urls[index], raw, base
        )
        completed += 1

        cookies = base.cookies.copy()

        def task(index):
            if stopped.is_set():
                raise CancelledError(
                    "下载停止，缓存保留"
                )

            if not hasattr(local, "session"):
                local.session = new_session(
                    cookies.copy()
                )

            return (
                index,
                download(
                    urls[index],
                    raw,
                    local.session,
                ),
            )

        pool = ThreadPoolExecutor(
            max_workers=workers
        )
        remaining = iter(missing[1:])

        def feed():
            while len(pending) < 2 * workers:
                index = next(
                    remaining, None
                )

                if index is None:
                    return

                future = pool.submit(
                    task, index
                )
                pending[future] = index

        feed()

        while pending:
            done, _ = wait(
                pending,
                timeout=0.5,
                return_when=FIRST_COMPLETED,
            )

            for future in done:
                pending.pop(future)
                index, path = future.result()
                existing[index] = path
                completed += 1

                print(
                    f"下载/校验 "
                    f"{completed}/{len(urls)}",
                    flush=True,
                )

            feed()

    except BaseException:
        stopped.set()

        for future in pending:
            future.cancel()

        raise

    finally:
        if pool is not None:
            pool.shutdown(
                wait=True,
                cancel_futures=True,
            )

        for session in sessions:
            session.close()

    save(
        raw / "pm10_download_report.json",
        {
            "files": len(urls),
            "downloaded": len(missing),
            "reused": (
                len(urls) - len(missing)
            ),
            "workers": workers,
            "seconds": (
                time.perf_counter() - started
            ),
        },
    )

    return existing


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--links",
        type=Path,
        default=DEFAULT_LINKS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_LINKS.parent / "merra_data"
        ),
    )
    parser.add_argument(
        "--year",
        type=int,
        help="不指定则依次处理清单中的全部年份",
    )
    parser.add_argument(
        "--workers",
        default="auto",
        help="auto或1至6，默认auto",
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="只用本地日文件，不登录、不下载",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        help=(
            "其他原始文件目录，"
            "须配合--year和--convert-only"
        ),
    )

    args = parser.parse_args()
    cores = available_cpu_count()

    try:
        workers = (
            automatic_workers(cores)
            if args.workers == "auto"
            else int(args.workers)
        )
    except ValueError:
        parser.error(
            "--workers必须为auto或1至6"
        )

    if not 1 <= workers <= 6:
        parser.error(
            "--workers必须为auto或1至6"
        )

    if args.raw_dir and (
        args.year is None
        or not args.convert_only
    ):
        parser.error(
            "--raw-dir必须配合"
            "--year和--convert-only"
        )

    print(
        f"启动：可用逻辑CPU {cores}，"
        f"下载并发 {workers}；"
        "逐时次计算，按年输出NetCDF4。",
        flush=True,
    )

    lists = {}

    for path in args.links.rglob(
        "*_links.txt"
    ):
        match = re.fullmatch(
            r"(\d{4})_links\.txt",
            path.name,
        )

        if match and (
            args.year is None
            or args.year == int(match[1])
        ):
            year = int(match[1])

            if year in lists:
                raise ValueError(
                    f"{year}有多个链接清单，"
                    "请指定唯一--links目录"
                )

            lists[year] = path

    if not lists:
        raise FileNotFoundError(
            "没有找到年度链接清单："
            f"{args.links}"
        )

    provinces = load_provinces(
        args.links / "CHN_ADM1.geojson"
    )

    credentials = None

    for year, link_file in sorted(
        lists.items()
    ):
        READ_WARNINGS.clear()

        urls, end = read_year_links(
            link_file,
            year,
            check_expiry=False,
        )

        fingerprint = sha256(
            (
                file_sha(link_file)
                + BOUNDARY_SHA
                + FORMULA
            ).encode()
        ).hexdigest()

        output = args.output / "pm10"

        if (
            output / f"{year}_pm10.nc4"
        ).exists():
            convert_year(
                [],
                year,
                output,
                provinces,
                end,
                fingerprint,
            )
            continue

        part = (
            output / f"{year}_pm10.nc4.part"
        )

        if part.exists():
            try:
                validate_output(
                    part,
                    year,
                    end,
                    fingerprint,
                )
            except (
                OSError,
                AttributeError,
                KeyError,
                ValueError,
                IndexError,
            ):
                pass
            else:
                convert_year(
                    [],
                    year,
                    output,
                    provinces,
                    end,
                    fingerprint,
                )
                continue

        raw = (
            args.raw_dir
            or args.output / "raw" / str(year)
        )

        if args.convert_only:
            files = (
                sorted(
                    path
                    for path in raw.iterdir()
                    if path.is_file()
                    and path.suffix.lower()
                    in (".nc", ".nc4")
                )
                if raw.is_dir()
                else []
            )

        else:
            raw.mkdir(
                parents=True,
                exist_ok=True,
            )

            files = []

            for number, url in enumerate(
                urls, 1
            ):
                files.append(
                    cached_file(url, raw)
                )

                if (
                    number == 1
                    or number % 30 == 0
                    or number == len(urls)
                ):
                    print(
                        f"{year} 检查原始缓存 "
                        f"{number}/{len(urls)}",
                        flush=True,
                    )

            if any(
                path is None
                for path in files
            ):
                read_year_links(
                    link_file,
                    year,
                    check_expiry=True,
                )

                if credentials is None:
                    credentials = (
                        input(
                            "Earthdata用户名："
                        ).strip(),
                        getpass(
                            "密码（不显示、不保存）："
                        ),
                    )

                files = fetch_missing(
                    urls,
                    raw,
                    files,
                    workers,
                    credentials,
                )

        print(
            f"{year} 开始计算PM10；"
            "原始日文件保留。",
            flush=True,
        )

        convert_year(
            files,
            year,
            output,
            provinces,
            end,
            fingerprint,
        )


if __name__ == "__main__":
    main()