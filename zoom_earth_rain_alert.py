"""Zoom Earth 降水覆盖检测（原型）

说明
- 该脚本使用 Selenium 打开 Zoom Earth 的降水图层页面，并截图。
- 在截图中计算“你家经纬度对应的屏幕像素点”，取该点周围一个小区域的平均颜色。
- 用“颜色是否足够偏蓝/青/紫”来粗略判断是否存在降水覆盖。

重要提醒（请务必了解）
1) Zoom Earth 的页面/数据来源和使用条款可能限制自动化抓取。请你确认自己使用符合其条款，仅个人学习/自用。
2) 该方法属于图像启发式判断：受缩放级别、底图颜色、图层透明度影响，误报/漏报都可能发生。
3) 更可靠的方案通常是直接获取合法的分钟级降水 API（如果能找到国内可直连的数据源）。

依赖
pip install selenium pillow requests

运行
- 需要本机安装 Chrome。
- 把企业微信机器人 webhook 填到 WEBHOOK。

"""

from __future__ import annotations

import io
import math
import time
# os 留着也行；方案B改为使用 config.py 读取 webhook
import os
from dataclasses import dataclass
from typing import Tuple
import colorsys
from datetime import datetime, timedelta, timezone

import requests
from PIL import Image

# 本地配置（不要上传/不要分享 config.py）
try:
    import config  # type: ignore
except Exception:
    config = None  # noqa

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service


# ----------------------- 配置区 -----------------------
# 建议用环境变量保存 webhook，避免写进代码泄露。
# Windows(PowerShell)：  setx WECOM_WEBHOOK "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx"
# 重新打开 PyCharm/终端后生效。
WEBHOOK = getattr(config, "WECOM_WEBHOOK", "") if config else ""  # 不设置则只打印不发送

# 你家的坐标  23.761781,121.474342
HOME_LAT = "*"
HOME_LON = "**"


# Zoom Earth 视图参数（缩放越大越精细，但加载更慢）
ZOOM = 11
MODEL = "icon"

# 预测多少分钟之后的天气
FORECAST_MINUTES = 15
# 每次点击时间轴后的等待时间
STEP_SLEEP_SEC = 0.8

# 颜色判定参数
# 目标：把“底图的绿/灰绿”排除掉，同时把“降水色带（蓝→绿→黄→橙→红→紫）”识别出来。
# 我们用 HSV 的饱和度 S 来做第一道过滤：
# - 底图通常饱和度低（灰绿/暗绿）
# - 降水色带通常饱和度高（明显的蓝/青/黄/红）
#
# 你希望“宁可多报一点”，所以只要落在降水色带且 S 足够高，就判定为雨。
SATURATION_THRESHOLD = 0.26  # 0~1，越大越严格
# 雨色带的 Hue 范围（度数 0~360）：
# - 蓝/青：约 170~260
# - 黄/橙/红：约 0~60
# - 紫：约 260~330（可选）
HUE_RANGES_DEG = [
    (170.0, 260.0),  # 蓝/青
    (0.0, 60.0),     # 黄/橙/红
    (260.0, 330.0),  # 紫
]

# 仍保留 score（用于调试），但最终判定不再用它做唯一依据
RAIN_SCORE_THRESHOLD = 110.0


# ----------------------- 工具函数 -----------------------

def send_wecom_text(webhook: str, content: str) -> None:
    if not webhook:
        print("[WARN] WEBHOOK 为空，只打印不发送：", content)
        return

    payload = {"msgtype": "text", "text": {"content": content}}
    r = requests.post(webhook, json=payload, timeout=10)
    r.raise_for_status()


@dataclass
class Viewport:
    width: int = 1280
    height: int = 720


def build_zoom_earth_url(lat: float, lon: float, zoom: int, model: str) -> str:
    """构建 Zoom Earth 的 URL。

    切换到 radar 图层：更贴近“实时雷达回波”。
    注意：radar 的时间轴与按钮是否支持“未来外推”不保证；如果只能回看历史，
    那我们就把它当成“实时是否正在下雨/附近有强回波”的检测。
    """
    # radar 页面不需要 model=icon，但保留参数不影响时可忽略
    return f"https://zoom.earth/maps/radar/#view={lat},{lon},{zoom}z"


def latlon_to_world_px(lat: float, lon: float, zoom: int) -> Tuple[float, float]:
    """Web Mercator: 经纬度 -> 世界像素坐标（zoom 对应的全世界像素平面）"""
    siny = math.sin(lat * math.pi / 180.0)
    # clamp
    siny = min(max(siny, -0.9999), 0.9999)

    scale = 256 * (2 ** zoom)
    x = (lon + 180.0) / 360.0 * scale
    y = (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)) * scale
    return x, y


def avg_color(img: Image.Image, center_xy: Tuple[int, int], radius: int = 6) -> Tuple[float, float, float]:
    cx, cy = center_xy
    x0 = max(cx - radius, 0)
    y0 = max(cy - radius, 0)
    x1 = min(cx + radius, img.width - 1)
    y1 = min(cy + radius, img.height - 1)

    region = img.crop((x0, y0, x1 + 1, y1 + 1)).convert("RGB")
    pixels = list(region.getdata())
    if not pixels:
        return 0.0, 0.0, 0.0
    r = sum(p[0] for p in pixels) / len(pixels)
    g = sum(p[1] for p in pixels) / len(pixels)
    b = sum(p[2] for p in pixels) / len(pixels)
    return r, g, b


def rgb_to_hsv_deg(rgb: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """RGB(0~255) -> HSV，其中 H 用度数(0~360)，S/V 用 0~1"""
    r, g, b = rgb
    r1, g1, b1 = (r / 255.0), (g / 255.0), (b / 255.0)
    h, s, v = colorsys.rgb_to_hsv(r1, g1, b1)  # h in [0,1)
    return h * 360.0, s, v


def hue_in_ranges(h_deg: float, ranges: list[tuple[float, float]]) -> bool:
    for a, b in ranges:
        if a <= b:
            if a <= h_deg <= b:
                return True
        else:
            # 支持跨 0 度的区间，例如 (350, 20)
            if h_deg >= a or h_deg <= b:
                return True
    return False


def is_rain_color(rgb: Tuple[float, float, float]) -> Tuple[bool, str]:
    """用 HSV 的“饱和度 + 色相范围”判断是否像降水色带（单像素）。"""
    h, s, v = rgb_to_hsv_deg(rgb)

    if s < SATURATION_THRESHOLD:
        return False, f"HSV=({h:.1f},{s:.2f},{v:.2f}) S<{SATURATION_THRESHOLD}"

    if not hue_in_ranges(h, HUE_RANGES_DEG):
        return False, f"HSV=({h:.1f},{s:.2f},{v:.2f}) H不在雨色带"

    return True, f"HSV=({h:.1f},{s:.2f},{v:.2f}) 命中雨色带"


def rain_coverage(img: Image.Image, center_xy: Tuple[int, int], radius: int = 12) -> Tuple[float, str]:
    """区域判定：统计中心点附近一个方形区域内，“命中雨色带”的像素比例。

    - radius=12 -> 区域大小约 (2*12+1)^2 = 625 像素
    - 返回: (命中比例 0~1, 调试信息)

    这样可以避免“单点恰好落在底图/文字/边界”导致的误判。
    """
    cx, cy = center_xy
    x0 = max(cx - radius, 0)
    y0 = max(cy - radius, 0)
    x1 = min(cx + radius, img.width - 1)
    y1 = min(cy + radius, img.height - 1)

    region = img.crop((x0, y0, x1 + 1, y1 + 1)).convert("RGB")
    pixels = list(region.getdata())
    if not pixels:
        return 0.0, "empty region"

    hit = 0
    sample_hsv = None
    for (r, g, b) in pixels:
        ok, _ = is_rain_color((float(r), float(g), float(b)))
        if ok:
            hit += 1
            if sample_hsv is None:
                h, s, v = rgb_to_hsv_deg((float(r), float(g), float(b)))
                sample_hsv = (h, s, v)

    ratio = hit / len(pixels)
    if sample_hsv:
        h, s, v = sample_hsv
        extra = f"sample_hit_HSV=({h:.1f},{s:.2f},{v:.2f})"
    else:
        extra = "sample_hit_HSV=None"

    return ratio, f"region={region.size} hit={hit}/{len(pixels)} ratio={ratio:.3f} {extra}"


def rain_score(rgb: Tuple[float, float, float]) -> float:
    """保留的调试分数（不作为唯一判定）。"""
    r, g, b = rgb
    warm = r
    purple = (r + b) * 0.6
    penalty_green = g * 0.9
    return warm + purple - penalty_green


def make_driver(viewport: Viewport) -> webdriver.Chrome:
    """创建 Edge WebDriver。

    你当前环境无法访问 storage.googleapis.com，因此不使用 webdriver-manager 自动下载 driver。
    这里改为使用你本机已下载的 msedgedriver.exe。
    """

    EDGEDRIVER_PATH = r"D:\\download\\edgedriver_win64 (1)\\msedgedriver.exe"  # 你提供的路径（Edge 143）

    options = webdriver.EdgeOptions()
    options.add_argument(f"--window-size={viewport.width},{viewport.height}")
    # 无头模式（你调试时可注释掉）
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = Service(executable_path=EDGEDRIVER_PATH)
    driver = webdriver.Edge(service=service, options=options)
    driver.set_page_load_timeout(60)
    return driver


class NextButtonNotClickable(Exception):
    """自定义异常，表示“下一个”按钮无法点击（例如被禁用）。"""
    pass


def get_map_time_tuple(driver: webdriver.Chrome) -> tuple[int, int] | None:
    """读取时间面板，并返回24小时制的 (小时, 分钟) 元组。"""
    try:
        hour_el = driver.find_element(By.CSS_SELECTOR, "div.panel.time.clock nav.clock-live div.hour div.text")
        minute_el = driver.find_element(By.CSS_SELECTOR, "div.panel.time.clock nav.clock-live div.minute div.text")
        ampm_el = driver.find_element(By.CSS_SELECTOR, "div.panel.time.clock nav.clock-live div.text.am-pm")

        hour_txt = (hour_el.text or "").strip()
        minute_txt = (minute_el.text or "").strip()
        ampm_txt = (ampm_el.text or "").strip()

        if not hour_txt or not minute_txt:
            return None

        hour = int(hour_txt)
        minute = int(minute_txt)

        # 转换成24小时制
        if "下午" in ampm_txt or "PM" in ampm_txt.upper():
            if hour < 12:
                hour += 12
        elif "上午" in ampm_txt or "AM" in ampm_txt.upper():
            if hour == 12:  # 上午12点是 0 点
                hour = 0

        return hour, minute

    except Exception:
        return None


def click_next(driver: webdriver.Chrome) -> str:
    """点击“分钟-下一个时间”按钮一次；如果不可点击则抛出 NextButtonNotClickable。"""

    css = "div.panel.time.clock nav.clock-live div.minute button.up[aria-label='下一个时间']"

    try:
        btn = driver.find_element(By.CSS_SELECTOR, css)
        # 如果 disabled，则说明已到达最新帧
        if btn.get_attribute("disabled") is not None:
            raise NextButtonNotClickable("next minute button is disabled")
    except NextButtonNotClickable:
        raise
    except Exception as e:
        raise NextButtonNotClickable(f"cannot find next minute button: {e!r}")

    # 即使存在也可能被遮挡/不可点击，仍用 wait 等一下
    try:
        wait = WebDriverWait(driver, 5)
        wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, css))).click()
    except Exception as e:
        raise NextButtonNotClickable(f"next minute button not clickable: {e!r}")

    time.sleep(STEP_SLEEP_SEC)
    return "minute.up"


def seek_to_target_time(driver: webdriver.Chrome, minutes_ahead: int) -> None:
    """把时间轴推进到“北京时间现在 + minutes_ahead”对应（或刚超过）的帧。"""
    bj_tz = timezone(timedelta(hours=8))
    target_dt = datetime.now(tz=bj_tz) + timedelta(minutes=minutes_ahead)
    target_hhmm = (target_dt.hour, target_dt.minute)

    def hhmm_ge(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return a[0] > b[0] or (a[0] == b[0] and a[1] >= b[1])

    print(f"[4/6] 目标时间=北京时间+{minutes_ahead}min => {target_hhmm[0]:02d}:{target_hhmm[1]:02d}")

    max_clicks = 60
    for i in range(max_clicks):
        hhmm = get_map_time_tuple(driver)
        print(f"    - 当前页面时间: {hhmm}")

        if hhmm and hhmm_ge(hhmm, target_hhmm):
            print(f"    - 已达到目标：{hhmm[0]:02d}:{hhmm[1]:02d} >= {target_hhmm[0]:02d}:{target_hhmm[1]:02d}")
            return

        try:
            btn_name = click_next(driver)
            print(f"    - click {btn_name} ({i + 1}/{max_clicks})")
        except NextButtonNotClickable as e:
            print(f"[WARN] 已到达最新可用时间，无法继续推进: {e!r}")
            return

    print("[WARN] 点击次数已到上限，仍未对齐目标时间。")


def main() -> None:
    print("SCRIPT_VERSION=2026-01-05-01")
    viewport = Viewport()
    url = build_zoom_earth_url(HOME_LAT, HOME_LON, ZOOM, MODEL)

    print("[1/6] 启动 Edge WebDriver...")
    driver = make_driver(viewport)
    try:
        print(f"[2/6] 打开页面: {url}")
        # 增加页面加载重试，应对 net::ERR_CONNECTION_CLOSED 这类网络抖动
        load_attempts = 3
        for i in range(load_attempts):
            try:
                driver.get(url)
                break  # 成功则跳出循环
            except Exception as e:
                print(f"    - 页面加载失败 (第 {i + 1}/{load_attempts} 次): {e!r}")
                if i < load_attempts - 1:
                    print("    - 3秒后重试...")
                    time.sleep(3)
                else:
                    print("[ERROR] 页面加载多次失败，请检查网络或网站可用性。")
                    raise

        # 你反馈：每次进入网站需要刷新一下，否则时间对不上。
        # 这里自动刷新一次。
        print("[2.5/6] 自动刷新一次页面（让时间轴对齐）...")
        driver.refresh()

        print("[3/6] 等待页面加载 8 秒...")
        time.sleep(8)

        # 推进到“北京时间 + FORECAST_MINUTES”
        seek_to_target_time(driver, FORECAST_MINUTES)

        print("[5/6] 截图并计算采样颜色...")
        png = driver.get_screenshot_as_png()
        img = Image.open(io.BytesIO(png))

        # 保存截图用于调试
        screenshot_path = "screenshot.png"
        img.save(screenshot_path)
        print(f"截图已保存为 {screenshot_path}，请打开图片确认屏幕中心点的颜色。")

        # 在固定 window-size 截图下，地图中心大致在屏幕中心。
        # 因为我们把 view 设置为 HOME 坐标，HOME 点≈屏幕中心
        sample_xy = (viewport.width // 2, viewport.height // 2)

        rgb = avg_color(img, sample_xy, radius=7)
        score = rain_score(rgb)

        # 区域覆盖判定（更稳）
        coverage, cov_dbg = rain_coverage(img, sample_xy, radius=12)
        ok = coverage >= 0.10  # 命中比例阈值：10%（可调）

        debug_msg = (
            f"ZoomEarth采样点(屏幕中心){sample_xy} "
            f"RGB={tuple(int(x) for x in rgb)} "
            f"score={score:.1f} (score_threshold={RAIN_SCORE_THRESHOLD}) "
            f"S_threshold={SATURATION_THRESHOLD} coverage={coverage:.3f} ({cov_dbg})"
        )
        print(debug_msg)

        print("[6/6] 生成结果...")
        if ok:
            msg = f"⚠️ 可能要下雨：你家附近检测到雷达回波色带覆盖。\n{debug_msg}\n{url}"
            print("[RESULT] RAIN_DETECTED")
            print(msg)
            send_wecom_text(WEBHOOK, msg)
        else:
            print("[RESULT] NO_RAIN_DETECTED")
            print("未检测到雷达回波色带覆盖（仅供参考）。")

    except Exception as e:
        print("[ERROR] 运行过程中出现异常：", repr(e))
        raise
    finally:
        print("[CLEANUP] 关闭浏览器")
        driver.quit()


if __name__ == "__main__":
    main()

