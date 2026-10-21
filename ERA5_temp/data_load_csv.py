"""读取已有年度链接，直接下载，再按年输出CSV和Parquet。"""

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
from urllib.parse import urlsplit

import netCDF4 as nc
import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import requests
from requests.auth import HTTPBasicAuth
from shapely import covers, points
from shapely.geometry import shape


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
    "3a00467a0db9b4136facb5f2f3d0edbfd"
    "96adb15651cfdf63991da9281030e85"
)

BOUNDARY_URL = (
    "https://media.githubusercontent.com/media/"
    "wmgeolab/geoBoundaries/"
    "5c25134028196d43ce97b5071934fd0cfc92f09f/"
    "releaseData/gbOpen/CHN/ADM1/"
    "geoBoundaries-CHN-ADM1.geojson"
)

DEFAULT_LINKS = (
    Path(r"D:\PythonProject\PythonProject\ERA5_temp\merra_links")
    if os.name == "nt"
    else Path(__file__).resolve().parent / "merra_links"
)

EDL_HOSTS = {
    "urs.earthdata.nasa.gov",
    "harmony.earthdata.nasa.gov",
    "opendap.earthdata.nasa.gov",
    "data.gesdisc.earthdata.nasa.gov",
}


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


def read_year_links(path, year, now=None, check_expiry=True):
    now = now or datetime.now(timezone.utc)
    urls = list(dict.fromkeys(
        line.strip()
        for line in path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ))

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
                "清单不是已核对的Harmony子集文件链接"
            )

        day = datetime.strptime(
            match[1], "%Y%m%d"
        ).date()

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
                "请保留本地数据，先刷新该年度结果"
            )

    dated.sort(key=lambda item: item[0])

    stop = datetime(
        last.year, last.month, last.day,
        21, tzinfo=timezone.utc,
    )

    return [url for _, url in dated], stop


class EarthdataSession(requests.Session):
    def __init__(self, username, password):
        super().__init__()
        self.credentials = HTTPBasicAuth(
            username, password
        )
        self.auth = self.authorize
        self.headers["Accept-Encoding"] = "identity"

    def authorize(self, request):
        parsed = urlsplit(request.url)
        request.headers.pop("Authorization", None)

        if parsed.scheme != "https":
            raise ValueError(
                "拒绝非HTTPS下载或重定向"
            )

        if parsed.hostname in EDL_HOSTS:
            self.credentials(request)

        return request

    def rebuild_auth(self, prepared_request, response):
        self.authorize(prepared_request)


def check_netcdf(path):
    with nc.Dataset(path) as ds:
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
    if urlsplit(url).scheme != "https":
        raise ValueError("链接必须是HTTPS")

    key = sha256(url.encode()).hexdigest()
    legacy = folder / f"{key[:24]}.nc4"

    day = re.search(
        r"\.(\d{8})_subsetted\.nc4$",
        urlsplit(url).path,
    )

    if (
        legacy.exists()
        or legacy.with_suffix(".part").exists()
    ):
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
                        "现有文件会保留，请先处理授权或刷新链接"
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
                        "无法安全按字节续传，未拼接文件"
                    )

                expected_size = None

                if response.status_code == 206:
                    match = re.fullmatch(
                        r"bytes (\d+)-(\d+)/(\d+)",
                        response.headers.get(
                            "Content-Range", ""
                        ),
                    )

                    if (
                        not match
                        or int(match[1]) != offset
                    ):
                        raise RuntimeError(
                            "续传范围不符，拒绝拼接文件"
                        )

                    expected_size = int(match[3])

                elif response.status_code == 200:
                    offset = 0
                    length = response.headers.get(
                        "Content-Length"
                    )
                    expected_size = (
                        int(length) if length else None
                    )

                else:
                    raise RuntimeError(
                        "非预期下载响应："
                        f"{response.status_code}"
                    )

                validator = response.headers.get("ETag")

                if validator and validator.startswith("W/"):
                    validator = None

                validator = (
                    validator
                    or response.headers.get("Last-Modified")
                )
                save(
                    resume_file,
                    {"validator": validator},
                )

                mode = "ab" if offset else "wb"

                with partial.open(mode) as stream:
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

            save(receipt, {
                "url_sha256": key,
                "sha256": file_sha(target),
                "bytes": target.stat().st_size,
            })

            resume_file.unlink(missing_ok=True)
            return target

        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ):
            if attempt == 2:
                raise RuntimeError(
                    "网络传输中断，部分文件已保留供下次续传"
                ) from None

    raise RuntimeError(
        "下载没有完成，部分文件已保留"
    )


def load_provinces(path):
    if not path.exists():
        print(
            "缓存中没有省界，获取原来使用的固定版本……",
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
            "省界文件哈希与获取链接时的固定版本不符"
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


def membership(lon, lat, provinces):
    xx, yy = np.meshgrid(lon, lat)

    masks = np.stack([
        covers(geometry, points(xx, yy))
        for geometry in provinces.values()
    ])

    selected = masks.any(axis=0)
    labels = np.full(
        selected.sum(), "", dtype=object
    )

    for name, mask in zip(provinces, masks):
        inside = mask[selected]
        labels[inside] = [
            str(old) + (";" if old else "") + name
            for old in labels[inside]
        ]

    return (
        selected,
        labels,
        xx[selected],
        yy[selected],
    )


def grid_ids(lon, lat):
    row = np.rint(
        (lat + 90) / 0.5
    ).astype(np.int64)

    col = np.rint(
        (lon + 180) / 0.625
    ).astype(np.int64)

    return row * 576 + col


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
        variable[:],
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

    if name == "lon":
        expected_units = {
            "degrees_east", "degree_east",
            "degrees_e", "degree_e",
        }
    else:
        expected_units = {
            "degrees_north", "degree_north",
            "degrees_n", "degree_n",
        }

    if coordinate_units not in expected_units:
        raise ValueError(
            f"{name}单位不是预期经纬度单位："
            f"{coordinate_units!r}"
        )

    values = np.asarray(
        ds[name][:], dtype=np.float64
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


def convert(
    files,
    year,
    output,
    provinces,
    formats=("csv", "parquet"),
    allow_partial=False,
    expected_stop=None,
):
    output.mkdir(parents=True, exist_ok=True)

    targets = {
        kind: output / f"{year}.{kind}"
        for kind in formats
    }

    if any(path.exists() for path in targets.values()):
        raise FileExistsError(
            "年度输出已存在；请保留原文件，"
            "换一个输出目录进行比较"
        )

    temporary = {
        kind: path.with_suffix(path.suffix + ".part")
        for kind, path in targets.items()
    }

    ordered = []

    for path in files:
        check_netcdf(path)

        with nc.Dataset(path) as ds:
            ordered.append(
                (min(file_times(ds)), path)
            )

    ordered.sort(
        key=lambda item: (item[0], str(item[1]))
    )

    world_lon = -180 + np.arange(576) * 0.625
    world_lat = -90 + np.arange(361) * 0.5

    _, _, expected_lon, expected_lat = membership(
        world_lon, world_lat, provinces
    )

    expected_grid = set(
        grid_ids(expected_lon, expected_lat)
    )

    if not expected_grid:
        raise ValueError("区域没有原始网格中心")

    seen = set()
    reference = None
    writers = {}
    buffer = []

    stats = {
        variable: {
            "missing": 0,
            "infinite": 0,
            "negative": 0,
            "nonpositive_density": 0,
            "mass_fraction_gt1": 0,
            "min": None,
            "max": None,
        }
        for variable in VARIABLES
    }

    rows = 0
    started = time.perf_counter()
    units = {}
    column_types = {}
    write_seconds = {
        kind: 0.0 for kind in formats
    }

    def flush():
        nonlocal rows, column_types

        if not buffer:
            return

        table = pa.concat_tables(buffer)

        column_types = {
            field.name: str(field.type)
            for field in table.schema
        }

        if not writers:
            if "csv" in formats:
                writers["csv"] = pacsv.CSVWriter(
                    str(temporary["csv"]),
                    table.schema,
                )

            if "parquet" in formats:
                writers["parquet"] = pq.ParquetWriter(
                    str(temporary["parquet"]),
                    table.schema,
                    compression="zstd",
                )

        for kind, writer in writers.items():
            before = time.perf_counter()
            writer.write_table(table)
            write_seconds[kind] += (
                time.perf_counter() - before
            )

        rows += len(table)
        buffer.clear()

    try:
        for _, path in ordered:
            with nc.Dataset(path) as ds:
                lon = axis_values(ds, "lon", 0.625)
                lat = axis_values(ds, "lat", 0.5)

                if reference is None:
                    reference = lon.copy(), lat.copy()

                elif (
                    not np.array_equal(reference[0], lon)
                    or not np.array_equal(reference[1], lat)
                ):
                    raise ValueError(
                        "不同文件的网格坐标不一致，"
                        "拒绝静默重采样"
                    )

                selected, labels, xx, yy = membership(
                    lon, lat, provinces
                )

                ids = grid_ids(xx, yy)
                count = len(xx)

                if set(ids) != expected_grid:
                    raise ValueError(
                        "文件未覆盖四省全部目标网格中心"
                    )

                for variable in VARIABLES:
                    actual = getattr(
                        ds[variable], "units", ""
                    ).strip()

                    expected = (
                        "kg m-3"
                        if variable == "AIRDENS"
                        else "kg kg-1"
                    )

                    if actual != expected:
                        raise ValueError(
                            f"{variable}单位不符：{actual!r}"
                        )

                    units[variable] = actual

                    if (
                        ds.data_model.startswith("NETCDF4")
                        and ds[variable].chunking()
                        != "contiguous"
                    ):
                        ds[variable].set_var_chunk_cache(
                            size=2 * 1024 * 1024,
                            nelems=1009,
                            preemption=0.75,
                        )

                for index, instant in enumerate(
                    file_times(ds)
                ):
                    if (
                        instant.year != year
                        or instant.hour % 3
                        or instant.minute
                        or instant.second
                        or instant.microsecond
                    ):
                        raise ValueError(
                            "文件时次不属于该年"
                            "或不在UTC三小时节点"
                        )

                    if instant in seen:
                        raise ValueError(
                            "文件之间存在重复时次，"
                            "请核对链接清单"
                        )

                    seen.add(instant)

                    timestamp = np.datetime64(
                        instant.replace(tzinfo=None), "ns"
                    )

                    data = {
                        "province": pa.array(labels.tolist()),
                        "grid_id": pa.array(ids),
                        "time_utc": pa.array(
                            np.full(count, timestamp),
                            type=pa.timestamp("ns", tz="UTC"),
                        ),
                        "longitude": pa.array(xx),
                        "latitude": pa.array(yy),
                        "model_level": pa.array(
                            np.full(count, 72, dtype=np.int8)
                        ),
                    }

                    for variable in VARIABLES:
                        source = ds[variable]
                        dims = source.dimensions

                        if (
                            not {"time", "lat", "lon"}.issubset(dims)
                            or set(dims)
                            - {"time", "lat", "lon", "lev"}
                        ):
                            raise ValueError(
                                f"{variable}维度不符：{dims}"
                            )

                        key = tuple(
                            index if dim == "time"
                            else 0 if dim == "lev"
                            else slice(None)
                            for dim in dims
                        )

                        values = np.ma.asarray(source[key])

                        spatial_dims = [
                            dim for dim in dims
                            if dim not in ("time", "lev")
                        ]

                        if spatial_dims == ["lon", "lat"]:
                            values = values.T

                        values = np.asarray(
                            values.filled(np.nan)
                        )[selected]

                        if values.dtype.kind != "f":
                            raise ValueError(
                                "变量不是预期浮点数据"
                            )

                        finite = values[
                            np.isfinite(values)
                        ]
                        item = stats[variable]

                        item["missing"] += int(
                            np.isnan(values).sum()
                        )
                        item["infinite"] += int(
                            np.isinf(values).sum()
                        )
                        item["negative"] += int(
                            (finite < 0).sum()
                        )

                        if variable == "AIRDENS":
                            item["nonpositive_density"] += int(
                                (finite <= 0).sum()
                            )
                        else:
                            item["mass_fraction_gt1"] += int(
                                (finite > 1).sum()
                            )

                        if finite.size:
                            minimum = float(finite.min())
                            maximum = float(finite.max())

                            item["min"] = (
                                min(item["min"], minimum)
                                if item["min"] is not None
                                else minimum
                            )

                            item["max"] = (
                                max(item["max"], maximum)
                                if item["max"] is not None
                                else maximum
                            )

                        data[variable] = pa.array(
                            values,
                            from_pandas=True,
                        )

                    buffer.append(pa.table(data))

                    if len(buffer) == 8:
                        flush()

        flush()

        times = sorted(seen)

        if not times or any(
            (second - first).total_seconds() != 10800
            for first, second in zip(
                times, times[1:]
            )
        ):
            raise ValueError(
                "年度时间序列存在缺口"
            )

        if expected_stop is not None and (
            times[0]
            != datetime(year, 1, 1, tzinfo=timezone.utc)
            or times[-1] != expected_stop
        ):
            raise ValueError(
                "NetCDF实际起止时次"
                "与该年链接日期范围不一致"
            )

        full_count = (
            366 if calendar.isleap(year) else 365
        ) * 8

        full = len(times) == full_count

        if not full and not allow_partial:
            raise ValueError(
                "不足完整一年；确认属于尚未发布完整"
                "的年份后，可用--allow-partial"
            )

        if rows != len(times) * len(expected_grid):
            raise ValueError(
                "输出行数与时次、网格数量不符"
            )

    except Exception:
        for writer in writers.values():
            writer.close()

        for path in temporary.values():
            path.unlink(missing_ok=True)

        raise

    for writer in writers.values():
        writer.close()

    for kind, path in targets.items():
        temporary[kind].replace(path)

    geometry_hash = sha256(
        b"".join(
            name.encode("utf-8") + geometry.wkb
            for name, geometry in provinces.items()
        )
    ).hexdigest()

    report = {
        "year": year,
        "rows": rows,
        "grid_count": len(expected_grid),
        "time_count": len(times),
        "dataset": "M2I3NVAER 5.12.4",
        "model_level": 72,
        "grid_spacing_degrees": {
            "longitude": 0.625,
            "latitude": 0.5,
        },
        "input_files": [
            {
                "filename": path.name,
                "sha256": file_sha(path),
            }
            for path in files
        ],
        "province_geometry_sha256": geometry_hash,
        "first_utc": times[0].isoformat(),
        "last_utc": times[-1].isoformat(),
        "complete_calendar_year": full,
        "elapsed_seconds_conversion": (
            time.perf_counter() - started
        ),
        "write_seconds": write_seconds,
        "column_types": column_types,
        "units": units,
        "variables": stats,
        "outputs": {
            kind: {
                "bytes": path.stat().st_size,
                "sha256": file_sha(path),
            }
            for kind, path in targets.items()
        },
        "boundary_rule": (
            "grid center covered by province polygon; "
            "shared centers have semicolon-separated names"
        ),
        "grid_id_definition": (
            "round((latitude+90)/0.5)*576"
            " + round((longitude+180)/0.625)"
        ),
        "note": (
            "原始数值精度保留；"
            "CSV类型、单位与UTC信息见此文件；"
            "未计算PM10"
        ),
    }

    save(
        output / f"{year}_metadata.json",
        report,
    )

    return report


def main():
    print(
        "程序已启动：读取年度链接，下载并整理年度数据。",
        flush=True,
    )

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
        default=DEFAULT_LINKS.parent / "merra_data",
    )
    parser.add_argument("--year", type=int)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        help="该年已有NetCDF文件的目录",
    )
    parser.add_argument(
        "--format",
        choices=("both", "csv", "parquet"),
        default="both",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--download-only",
        action="store_true",
    )
    mode.add_argument(
        "--convert-only",
        action="store_true",
    )

    args = parser.parse_args()

    if args.raw_dir is not None and (
        not args.convert_only
        or args.year is None
    ):
        parser.error(
            "--raw-dir须与--convert-only"
            "和--year一起使用"
        )

    lists = {}

    for path in args.links.rglob("*_links.txt"):
        match = re.fullmatch(
            r"(\d{4})_links\.txt",
            path.name,
        )

        if match and (
            args.year is None
            or int(match[1]) == args.year
        ):
            year = int(match[1])

            if year in lists:
                raise ValueError(
                    "同一年存在多个链接清单，"
                    "请明确--links目录"
                )

            lists[year] = path

    if not lists:
        raise FileNotFoundError(
            f"未找到年度链接清单：{args.links}"
        )

    provinces = load_provinces(
        args.links / "CHN_ADM1.geojson"
    )

    if args.convert_only:
        session = None
    else:
        username = input(
            "Earthdata用户名："
        ).strip()
        password = getpass(
            "密码（不显示）："
        )
        session = EarthdataSession(
            username, password
        )

    for year, link_file in sorted(lists.items()):
        urls, expected_stop = read_year_links(
            link_file,
            year,
            check_expiry=not args.convert_only,
        )

        print(
            f"{year}：{len(urls)}个日文件，"
            f"清单截至{expected_stop.date()}；"
            "最终按年输出",
            flush=True,
        )

        raw = (
            args.raw_dir
            if args.raw_dir is not None
            else args.output / "raw" / str(year)
        )

        if args.convert_only:
            files = (
                sorted(
                    path
                    for path in raw.iterdir()
                    if path.is_file()
                    and path.suffix.lower() in (
                        ".nc", ".nc4"
                    )
                )
                if raw.is_dir()
                else []
            )

            if not files:
                raise FileNotFoundError(
                    f"没有找到NetCDF文件：{raw}"
                )

        else:
            raw.mkdir(
                parents=True,
                exist_ok=True,
            )
            files = []
            started = time.perf_counter()

            for number, url in enumerate(urls, 1):
                print(
                    f"{year} 下载/校验 "
                    f"{number}/{len(urls)}",
                    flush=True,
                )
                files.append(
                    download(url, raw, session)
                )

            print(
                f"{year} 下载/校验耗时 "
                f"{time.perf_counter() - started:.1f}秒",
                flush=True,
            )

        if args.download_only:
            continue

        print(
            f"{year} 开始年度转换",
            flush=True,
        )

        formats = (
            ("csv", "parquet")
            if args.format == "both"
            else (args.format,)
        )

        report = convert(
            files,
            year,
            args.output / "processed",
            provinces,
            formats,
            args.allow_partial
            or year == datetime.now(timezone.utc).year,
            expected_stop,
        )

        print(
            json.dumps(
                report["outputs"],
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

        print(
            f"完成：{report['rows']}行，"
            f"{report['grid_count']}网格，"
            f"{report['time_count']}时次",
            flush=True,
        )


if __name__ == "__main__":
    main()