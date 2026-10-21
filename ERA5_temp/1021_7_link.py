"""获取7个邻国2015年至最新时次的MERRA-2年度子集链接。"""

from datetime import datetime, timedelta, timezone
from getpass import getpass
from hashlib import sha256
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import urlopen

from harmony import Client, Collection, Dimension, Request
from harmony.client import ProcessingFailedException
from shapely.geometry import Point, mapping, shape


# ==================== 配置 ====================

START_YEAR = 2015

# None表示处理到NASA目录最新可用年份。
END_YEAR = None

# 本脚本放在：
# D:\PythonProject\PythonProject\ERA5_temp
BASE = Path(__file__).resolve().parent

OUT = BASE / "merra_links_neighbors"

COLLECTION_ID = "C1276812882-GES_DISC"

VARIABLES = [
    "AIRDENS",
    "SO4",
    "BCPHOBIC",
    "BCPHILIC",
    "OCPHOBIC",
    "OCPHILIC",
    "DU001",
    "DU002",
    "DU003",
    "DU004",
    "SS001",
    "SS002",
    "SS003",
    "SS004",
]

# 坐标只用于核查选中的国界，不用于提取城市数据。
COUNTRIES = {
    "蒙古国": ("Mongolia", (106.917, 47.918)),
    "哈萨克斯坦": ("Kazakhstan", (71.43, 51.13)),
    "吉尔吉斯斯坦": ("Kyrgyzstan", (74.59, 42.87)),
    "塔吉克斯坦": ("Tajikistan", (68.78, 38.56)),
    "阿富汗": ("Afghanistan", (69.17, 34.53)),
    "巴基斯坦": ("Pakistan", (73.05, 33.69)),
    "印度": ("India", (77.21, 28.61)),
}

BOUNDARY_URL = (
    "https://raw.githubusercontent.com/"
    "nvkelso/natural-earth-vector/"
    "v5.1.2/geojson/"
    "ne_50m_admin_0_countries.geojson"
)

BOUNDARY_SHA = (
    "3e458fc036ad0a66411f2c1e6cac49c5d"
    "7bfb81cb1123bc513b22511a2b7fdeb"
)


# ==================== 基础工具 ====================

def fetch(url):
    """临时网络错误最多尝试3次。"""
    for attempt in range(3):
        try:
            with urlopen(url, timeout=60) as response:
                return response.read()

        except HTTPError as exc:
            if (
                exc.code not in (429, 500, 502, 503, 504)
                or attempt == 2
            ):
                raise

        except (URLError, TimeoutError):
            if attempt == 2:
                raise

        print("临时网络错误，稍后重试……", flush=True)
        time.sleep(2 ** attempt)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


def file_sha(path):
    return sha256(path.read_bytes()).hexdigest()


def as_utc(value):
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )

    if result.tzinfo is None:
        raise ValueError("NASA时间缺少时区。")

    return result.astimezone(timezone.utc)


# ==================== 准备国家边界 ====================

def prepare_regions():
    folder = OUT / "boundaries"
    folder.mkdir(parents=True, exist_ok=True)

    source = folder / "ne_50m_admin_0_countries_v5.1.2.geojson"

    if source.exists():
        raw = source.read_bytes()
    else:
        print(
            "首次运行，正在下载固定版本国界……",
            flush=True,
        )
        raw = fetch(BOUNDARY_URL)

    if sha256(raw).hexdigest() != BOUNDARY_SHA:
        raise ValueError(
            "国界文件校验失败，未覆盖已有文件。"
        )

    if not source.exists():
        source.write_bytes(raw)

    features = json.loads(raw)["features"]
    regions = {}

    for chinese, (english, capital) in COUNTRIES.items():
        matches = [
            feature
            for feature in features
            if feature["properties"].get("ADMIN") == english
        ]

        if len(matches) != 1:
            raise ValueError(
                f"{chinese}国界未唯一匹配。"
            )

        geometry = shape(matches[0]["geometry"])

        if (
            geometry.geom_type not in ("Polygon", "MultiPolygon")
            or geometry.is_empty
            or not geometry.is_valid
            or not geometry.covers(Point(capital))
        ):
            raise ValueError(
                f"{chinese}国界检查失败。"
            )

        # 保留数据集中的全部多边形，不只选最大多边形。
        region_file = folder / f"{english}.geojson"

        save_json(
            region_file,
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "country": chinese,
                            "country_en": english,
                            "source": (
                                "Natural Earth v5.1.2, 1:50m"
                            ),
                            "source_sha256": BOUNDARY_SHA,
                        },
                        "geometry": mapping(geometry),
                    }
                ],
            },
        )

        regions[chinese] = region_file

        print(
            f"{chinese}：国界检查通过，"
            f"范围为{geometry.bounds}",
            flush=True,
        )

    return regions


# ==================== 查询最新时间 ====================

def latest_available():
    query = urlencode(
        {
            "collection_concept_id": COLLECTION_ID,
            "sort_key[]": "-start_date",
            "page_size": 1,
        }
    )

    catalog = json.loads(
        fetch(
            "https://cmr.earthdata.nasa.gov/"
            "search/granules.umm_json?"
            + query
        )
    )

    if not catalog.get("items"):
        raise RuntimeError(
            "CMR没有返回数据目录，未提交任务。"
        )

    extent = catalog["items"][0]["umm"]["TemporalExtent"]
    interval = extent.get("RangeDateTime", {})

    latest_text = (
        interval.get("EndingDateTime")
        or extent.get("SingleDateTime")
    )

    if not latest_text:
        raise RuntimeError(
            "CMR缺少最新结束时间，未提交任务。"
        )

    latest = min(
        as_utc(latest_text),
        datetime.now(timezone.utc),
    )

    latest = latest.replace(
        hour=latest.hour // 3 * 3,
        minute=0,
        second=0,
        microsecond=0,
    )

    return latest


def year_window(year, latest):
    start = datetime(
        year, 1, 1, tzinfo=timezone.utc
    )

    stop = min(
        datetime(
            year + 1, 1, 1, tzinfo=timezone.utc
        ) - timedelta(hours=3),
        latest,
    )

    return start, stop


# ==================== 构造NASA请求 ====================

def make_request(region_file, start, stop):
    request = Request(
        collection=Collection(COLLECTION_ID),
        shape=str(region_file),
        temporal={
            "start": start,
            "stop": (
                stop
                if stop > start
                else stop + timedelta(seconds=1)
            ),
        },
        variables=VARIABLES,
        dimensions=[
            Dimension(
                name="lev",
                min=72,
                max=72,
            )
        ],
        format="application/netcdf",
        skip_preview=True,
    )

    if not request.is_valid():
        raise ValueError(
            "; ".join(request.error_messages())
        )

    return request


# ==================== 任务编号兼容与恢复 ====================

def normalize_job_id(value):
    """
    接受：
    1. 正常UUID编号；
    2. SDK返回的完整状态字典；
    3. 旧程序误存的状态字典字符串。
    """
    import ast
    from uuid import UUID

    def canonical(item):
        candidate = str(item).strip()

        try:
            identifier = str(UUID(candidate))
        except (ValueError, TypeError, AttributeError):
            return None

        if candidate.lower() == identifier:
            return identifier

        return None

    direct = canonical(value)

    if direct:
        return direct

    payload = value

    if isinstance(value, str):
        if len(value) > 8 * 1024 * 1024:
            raise ValueError(
                "任务记录超过安全解析长度，"
                "未发送状态请求。"
            )

        try:
            payload = json.loads(value)
        except (ValueError, TypeError):
            try:
                # 只解析Python字面量，不执行其中的代码。
                payload = ast.literal_eval(value)
            except (
                ValueError,
                SyntaxError,
                RecursionError,
            ) as exc:
                raise ValueError(
                    "job_id不是合法UUID或可恢复的状态记录；"
                    "未发送请求。"
                ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            "无法识别Harmony返回的任务编号类型；"
            "未发送请求。"
        )

    candidates = set()

    for key in ("jobID", "job_id", "jobId", "id"):
        identifier = canonical(payload.get(key))

        if identifier:
            candidates.add(identifier)

    uuid_pattern = (
        r"[0-9a-fA-F]{8}"
        r"(?:-[0-9a-fA-F]{4}){3}"
        r"-[0-9a-fA-F]{12}"
    )

    links = payload.get("links", [])

    if not isinstance(links, list):
        raise ValueError(
            "状态记录的links类型异常，未发送请求。"
        )

    for link in links:
        if not isinstance(link, dict):
            continue

        parsed = urlsplit(
            str(link.get("href", ""))
        )

        if (
            parsed.scheme != "https"
            or parsed.hostname
            != "harmony.earthdata.nasa.gov"
        ):
            continue

        for pattern in (
            rf"^/(?:jobs|stac)/({uuid_pattern})(?:/|$)",
            rf"^/service-results/.+?/public/"
            rf"({uuid_pattern})(?:/|$)",
        ):
            match = re.search(pattern, parsed.path)

            if match:
                candidates.add(
                    str(UUID(match[1]))
                )

    if len(candidates) != 1:
        raise ValueError(
            "无法确定唯一任务编号，"
            f"发现{len(candidates)}个候选。"
            "请保留任务记录，不要重复提交。"
        )

    return candidates.pop()


def ensure_job_id(state_file, state):
    """在查询前恢复旧记录，并备份修改前的原文件。"""
    import shutil

    old = state.get("job_id")
    identifier = normalize_job_id(old)

    if identifier != old:
        if state_file.exists():
            stamp = datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%S%fZ"
            )

            backup = state_file.with_name(
                f"{state_file.stem}"
                f"_bad_job_id_{stamp}.json"
            )

            shutil.copy2(state_file, backup)

            print(
                f"原任务记录已备份：{backup}",
                flush=True,
            )

        state["job_id"] = identifier
        save_json(state_file, state)

        print(
            f"已恢复合法任务编号：{identifier}"
            "（未重新提交）",
            flush=True,
        )

    return identifier


# ==================== 管理任务状态 ====================

def update_status(client, state_file, state):
    ensure_job_id(state_file, state)

    info = client.status(state["job_id"])

    for key in (
        "status",
        "message",
        "errors",
        "data_expiration",
        "progress",
    ):
        if key in info:
            state[key] = info[key]

    save_json(state_file, state)

    return info


def wait_success(client, state_file, state):
    ensure_job_id(state_file, state)

    try:
        client.wait_for_processing(
            state["job_id"],
            show_progress=True,
        )

    except ProcessingFailedException:
        update_status(
            client,
            state_file,
            state,
        )

        raise RuntimeError(
            "NASA任务失败，错误详情已保存到："
            f"{state_file}。"
            "请先检查errors，修复原因后重新运行。"
        ) from None

    info = update_status(
        client,
        state_file,
        state,
    )

    if info.get("status") != "successful":
        raise RuntimeError(
            "NASA任务未完整成功，请查看："
            f"{state_file}"
        )

    return info


def expired(info):
    expiry = info.get("data_expiration")

    return bool(
        expiry
        and as_utc(expiry)
        <= datetime.now(timezone.utc)
    )


def archive_old(
    folder,
    year,
    state_file,
    target,
):
    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    history = (
        folder
        / "_history"
        / f"{year}_{stamp}"
    )

    history.mkdir(
        parents=True,
        exist_ok=False,
    )

    for path in (state_file, target):
        if path.exists():
            path.replace(history / path.name)

    print(
        "旧任务记录及清单已归档："
        f"{history}",
        flush=True,
    )


# ==================== 检查链接 ====================

def validate_links(urls, year, start, stop):
    if not urls:
        raise RuntimeError(
            "NASA未返回结果链接。"
        )

    dates = []

    for url in urls:
        parsed = urlsplit(url)

        if parsed.scheme != "https":
            raise ValueError(
                "NASA返回非HTTPS结果链接，"
                "未写入清单。"
            )

        match = re.search(
            r"\.(\d{8})(?:_subsetted)?\.nc4?$",
            parsed.path,
        )

        if match:
            day = datetime.strptime(
                match[1], "%Y%m%d"
            ).date()

            if day.year != year:
                raise ValueError(
                    "NASA结果包含其他年份，"
                    "未写入清单。"
                )

            dates.append(day)

    if len(dates) == len(urls):
        expected = {
            start.date() + timedelta(days=index)
            for index in range(
                (stop.date() - start.date()).days + 1
            )
        }

        if (
            len(dates) != len(set(dates))
            or set(dates) != expected
        ):
            raise ValueError(
                "结果链接日期有缺失或重复，"
                "未写入清单。"
            )

    else:
        print(
            "提示：部分文件名无法解析日期，"
            "已记录链接数量，未验证日完整性。",
            flush=True,
        )


# ==================== 处理一个国家的一年 ====================

def process_year(
    client,
    country,
    region_file,
    year,
    latest,
):
    start, stop = year_window(year, latest)

    decade = year // 10 * 10

    folder = (
        OUT
        / country
        / f"{decade}-{decade + 9}"
    )

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    state_file = folder / f"{year}_job.json"
    target = folder / f"{year}_links.txt"

    # 不含截止时间的稳定配置。
    config = {
        "country": country,
        "collection": COLLECTION_ID,
        "variables": VARIABLES,
        "level": 72,
        "region_sha256": file_sha(region_file),
        "start": start.isoformat(),
    }

    config_hash = sha256(
        json.dumps(
            config,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    identity = dict(
        config,
        stop=stop.isoformat(),
    )

    fingerprint = sha256(
        json.dumps(
            identity,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    state = None

    if state_file.exists():
        state = json.loads(
            state_file.read_text(
                encoding="utf-8"
            )
        )

        if state.get("config_hash") != config_hash:
            raise RuntimeError(
                f"{country} {year}请求配置变化。"
                "请保留旧记录，改用新的OUT目录。"
            )

        if not state.get("job_id"):
            raise RuntimeError(
                "上次提交是否成功不明："
                f"{state_file}。"
                "请先核对NASA任务，"
                "避免重复提交。"
            )

        # 此处会先自动修复旧的异常job_id。
        info = update_status(
            client,
            state_file,
            state,
        )

        changed = (
            state.get("request_hash")
            != fingerprint
        )

        invalid = (
            expired(info)
            or info.get("status") in (
                "failed",
                "complete_with_errors",
                "canceled",
                "cancelled",
                "expired",
                "deleted",
            )
        )

        if changed or invalid:
            if (
                changed
                and not invalid
                and info.get("status") != "successful"
            ):
                print(
                    f"{country} {year}"
                    "旧任务仍在处理，"
                    "先等待结束，再更新截止时间。",
                    flush=True,
                )

                wait_success(
                    client,
                    state_file,
                    state,
                )

            archive_old(
                folder,
                year,
                state_file,
                target,
            )

            state = None

        elif (
            info.get("status") == "successful"
            and target.exists()
        ):
            if (
                state.get("links_sha256")
                == file_sha(target)
            ):
                print(
                    f"{country} {year}："
                    "任务成功且清单校验通过，跳过。",
                    flush=True,
                )
                return

    elif target.exists():
        raise RuntimeError(
            "清单存在但任务记录缺失，"
            f"未覆盖：{target}"
        )

    if state is None:
        request = make_request(
            region_file,
            start,
            stop,
        )

        state = dict(
            identity,
            year=year,
            config_hash=config_hash,
            request_hash=fingerprint,
            status="submitting",
        )

        # 提交前保存记录，避免中断后盲目重复提交。
        save_json(state_file, state)

        submit_result = client.submit(request)

        try:
            state["job_id"] = normalize_job_id(
                submit_result
            )

        except ValueError:
            response_file = (
                folder
                / f"{year}_submit_response.json"
            )

            save_json(
                response_file,
                {"response": submit_result},
            )

            raise RuntimeError(
                "提交返回的任务编号无法解析，"
                "原始返回已保存："
                f"{response_file}。"
                "请保留记录，避免重复提交。"
            ) from None

        state["status"] = "submitted"

        save_json(state_file, state)

        print(
            f"{country} {year}："
            "已提交年度任务 "
            f"{state['job_id']}",
            flush=True,
        )

    else:
        print(
            f"{country} {year}："
            f"恢复任务 {state['job_id']}",
            flush=True,
        )

    info = wait_success(
        client,
        state_file,
        state,
    )

    if expired(info):
        raise RuntimeError(
            "任务结果已过期，未写入链接："
            f"{state_file}。"
            "重新运行将归档旧记录后更新。"
        )

    # 不设置page或limit，保留SDK遍历全部结果页的行为。
    urls = list(
        dict.fromkeys(
            client.result_urls(
                state["job_id"]
            )
        )
    )

    urls.sort()

    validate_links(
        urls,
        year,
        start,
        stop,
    )

    temporary = target.with_suffix(".txt.tmp")

    temporary.write_text(
        "\n".join(urls) + "\n",
        encoding="utf-8",
    )

    temporary.replace(target)

    state["link_count"] = len(urls)
    state["links_sha256"] = file_sha(target)

    state["links_written_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )

    save_json(state_file, state)

    print(
        f"{country} {year}："
        f"已保存{len(urls)}条链接\n"
        f"{target}",
        flush=True,
    )

    if state.get("data_expiration"):
        print(
            "NASA结果到期时间（UTC）："
            f"{state['data_expiration']}",
            flush=True,
        )


# ==================== 主程序 ====================

def main():
    print("=" * 60, flush=True)
    print(
        "启动：获取7国MERRA-2年度链接",
        flush=True,
    )
    print(
        "本脚本不下载气溶胶文件，不计算PM10。",
        flush=True,
    )
    print("=" * 60, flush=True)

    if (
        not 1980 <= START_YEAR <= 9998
        or (
            END_YEAR is not None
            and not START_YEAR <= END_YEAR <= 9998
        )
    ):
        raise ValueError("年份范围错误。")

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("准备7国多边形……", flush=True)
    regions = prepare_regions()

    print(
        "查询NASA最新可用时间……",
        flush=True,
    )

    latest = latest_available()

    if START_YEAR > latest.year:
        raise ValueError(
            "开始年份超过最新实际可用年份。"
        )

    end_year = (
        min(END_YEAR, latest.year)
        if END_YEAR is not None
        else latest.year
    )

    print(
        f"处理年份：{START_YEAR}—{end_year}",
        flush=True,
    )
    print(
        "NASA目录最新时次："
        f"{latest.isoformat()}",
        flush=True,
    )
    print(
        f"链接保存目录：{OUT}",
        flush=True,
    )

    # 每次运行输入一次，所有国家和年份复用这个Client。
    username = input(
        "Earthdata用户名："
    ).strip()

    password = getpass(
        "Earthdata密码（不显示、不写入文件）："
    )

    if not username or not password:
        raise ValueError(
            "用户名和密码不能为空。"
        )

    client = Client(
        auth=(username, password)
    )

    for country, region_file in regions.items():
        for year in range(
            START_YEAR,
            end_year + 1,
        ):
            print(
                f"\n===== {country} {year} =====",
                flush=True,
            )

            process_year(
                client,
                country,
                region_file,
                year,
                latest,
            )

    print(
        "\n全部链接获取完成；"
        "本脚本没有下载MERRA气溶胶文件。",
        flush=True,
    )


if __name__ == "__main__":
    main()