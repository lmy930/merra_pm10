from datetime import datetime, timedelta, timezone
from getpass import getpass
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import json

from harmony import Client, Collection, Dimension, Request
from harmony.client import ProcessingFailedException
from shapely.geometry import Point, mapping, shape
from shapely.ops import unary_union


# ===== 修改这里 =====
START_YEAR = 1980
END_YEAR = None  # None：截至最新实际可用年份；填1989则只处理到1989年
OUT = Path(__file__).resolve().parent / "merra_links"

COLLECTION = "C1276812882-GES_DISC"

VARIABLES = [
    "AIRDENS", "SO4",
    "BCPHOBIC", "BCPHILIC",
    "OCPHOBIC", "OCPHILIC",
    "DU001", "DU002", "DU003", "DU004",
    "SS001", "SS002", "SS003", "SS004",
]

# 名称已经按实际源文件核对。
# 坐标只用于检查省界是否选对，不用于提取城市数据。
PROVINCES = {
    "宁夏": (
        "Ningxia Ningxia Hui Autonomous Region",
        (106.23, 38.49),
    ),
    "新疆": (
        "Xinjiang Uyghur Autonomous Region",
        (87.62, 43.82),
    ),
    "甘肃": (
        "Gansu Province",
        (103.83, 36.06),
    ),
    "内蒙古": (
        "Inner Mongolia Autonomous Region",
        (111.75, 40.84),
    ),
}

BOUNDARY_URL = (
    "https://media.githubusercontent.com/media/"
    "wmgeolab/geoBoundaries/"
    "5c25134028196d43ce97b5071934fd0cfc92f09f/"
    "releaseData/gbOpen/CHN/ADM1/"
    "geoBoundaries-CHN-ADM1.geojson"
)

BOUNDARY_SHA = (
    "3a00467a0db9b4136facb5f2f3d0edbfd"
    "96adb15651cfdf63991da9281030e85"
)


def fetch(url):
    with urlopen(url, timeout=60) as response:
        return response.read()


def save(path, value):
    temporary = path.with_suffix(".tmp")
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


def prepare_region():
    """将四个完整省界合成一个联合区域。"""
    OUT.mkdir(parents=True, exist_ok=True)
    cached = OUT / "CHN_ADM1.geojson"

    raw = (
        cached.read_bytes()
        if cached.exists()
        else fetch(BOUNDARY_URL)
    )

    if sha256(raw).hexdigest() != BOUNDARY_SHA:
        raise ValueError("源省界校验失败。")

    features = json.loads(raw)["features"]
    polygons = []

    for province, (name, point) in PROVINCES.items():
        matches = [
            feature
            for feature in features
            if feature["properties"]["shapeName"] == name
        ]

        if len(matches) != 1:
            raise ValueError(f"{province}未唯一匹配。")

        polygon = shape(matches[0]["geometry"])

        if (
            polygon.geom_type not in ("Polygon", "MultiPolygon")
            or not polygon.is_valid
            or not polygon.covers(Point(point))
        ):
            raise ValueError(f"{province}边界检查失败。")

        polygons.append(polygon)

    region = unary_union(polygons)

    if not region.is_valid or region.is_empty:
        raise ValueError("四省联合边界无效。")

    cached.write_bytes(raw)
    region_file = OUT / "four_provinces.geojson"

    save(region_file, {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"provinces": list(PROVINCES)},
            "geometry": mapping(region),
        }],
    })

    return region_file


def year_window(year, latest):
    start = datetime(year, 1, 1, tzinfo=timezone.utc)

    if start > latest:
        return None

    stop = min(
        datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        - timedelta(hours=3),
        latest,
    )

    return start, stop


def make_request(region_file, start, stop):
    if start == stop:
        stop += timedelta(seconds=1)

    request = Request(
        collection=Collection(COLLECTION),
        shape=str(region_file),
        temporal={"start": start, "stop": stop},
        variables=VARIABLES,
        dimensions=[
            Dimension(name="lev", min=72, max=72)
        ],
        format="application/netcdf",
        skip_preview=True,
    )

    if not request.is_valid():
        raise ValueError("; ".join(request.error_messages()))

    return request


def main():
    if (
        not 1980 <= START_YEAR <= 9998
        or (
            END_YEAR is not None
            and not START_YEAR <= END_YEAR <= 9998
        )
    ):
        raise ValueError("年份范围不正确。")

    username = input("Earthdata用户名：").strip()
    password = getpass("Earthdata密码（不显示）：")
    client = Client(auth=(username, password))

    print("准备四省联合边界……", flush=True)
    region_file = prepare_region()

    query = urlencode({
        "collection_concept_id": COLLECTION,
        "sort_key[]": "-start_date",
        "page_size": 1,
    })

    catalog = json.loads(fetch(
        "https://cmr.earthdata.nasa.gov/"
        "search/granules.umm_json?" + query
    ))

    latest_text = (
        catalog["items"][0]["umm"]["TemporalExtent"]
        ["RangeDateTime"]["EndingDateTime"]
    )
    latest = datetime.fromisoformat(
        latest_text.replace("Z", "+00:00")
    )

    print("最新目录时次：", latest.isoformat(), flush=True)

    if START_YEAR > latest.year:
        raise ValueError("开始年份超过最新实际可用年份。")

    end_year = (
        min(END_YEAR, latest.year)
        if END_YEAR is not None
        else latest.year
    )

    # 每年一次请求，不再循环省份和月份。
    for year in range(START_YEAR, end_year + 1):
        start, stop = year_window(year, latest)

        decade = year // 10 * 10
        folder = OUT / f"{decade}-{decade + 9}"
        folder.mkdir(exist_ok=True)

        state_file = folder / f"{year}_job.json"
        target = folder / f"{year}_links.txt"

        identity = json.dumps([
            COLLECTION,
            VARIABLES,
            72,
            sha256(region_file.read_bytes()).hexdigest(),
            start.isoformat(),
            stop.isoformat(),
        ])
        fingerprint = sha256(identity.encode()).hexdigest()

        if state_file.exists():
            state = json.loads(
                state_file.read_text(encoding="utf-8")
            )

            if state.get("request_hash") != fingerprint:
                raise RuntimeError(
                    f"{year}请求范围已变化。"
                    "请保留旧记录，改用新的OUT目录。"
                )

            if not state.get("job_id"):
                raise RuntimeError(
                    f"{year}上次提交结果不明，"
                    "请先核对NASA任务，避免重复提交。"
                )

            job_id = state["job_id"]
            print(
                year, "查询已有年度任务：", job_id,
                flush=True,
            )

        else:
            request = make_request(region_file, start, stop)

            state = {
                "year": year,
                "request_hash": fingerprint,
                "status": "submitting",
            }
            save(state_file, state)

            job_id = str(client.submit(request))

            state.update(
                job_id=job_id,
                status="submitted",
            )
            save(state_file, state)

            print(
                year, "四省联合年度任务已提交：", job_id,
                flush=True,
            )

        try:
            client.wait_for_processing(
                job_id,
                show_progress=True,
            )

        except ProcessingFailedException:
            info = client.status(job_id)

            state.update({
                key: info.get(key)
                for key in ("status", "message", "errors")
            })
            save(state_file, state)

            raise RuntimeError(
                f"{year}年度任务失败，"
                f"具体错误已保存到：{state_file}。"
                "程序没有重新提交，也没有改成按月任务。"
            ) from None

        info = client.status(job_id)

        state.update({
            key: info.get(key)
            for key in (
                "status", "message",
                "errors", "data_expiration",
            )
        })
        save(state_file, state)

        if info["status"] != "successful":
            raise RuntimeError(
                f"{year}任务未完整成功，"
                f"请查看：{state_file}"
            )

        links = list(dict.fromkeys(
            client.result_urls(job_id)
        ))

        if not links:
            raise RuntimeError(f"{year}未返回文件链接。")

        temporary = target.with_suffix(".tmp")
        temporary.write_text(
            "\n".join(links) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)

        print(
            f"{year}：四省合计{len(links)}条结果链接，"
            f"已保存到 {target}",
            flush=True,
        )

    print("链接生成结束，Python未下载MERRA气溶胶数据。")


if __name__ == "__main__":
    main()