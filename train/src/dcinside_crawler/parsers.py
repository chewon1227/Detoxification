import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from .config import CrawlConfig
from .models import Post


def parse_search_items(page_source: str) -> List[Post]:
    soup = BeautifulSoup(page_source, "lxml")
    rows: List[Post] = []
    for li in soup.select(".sch_result_list li"):
        link_el = li.select_one("a.tit_txt")
        date_el = li.select_one("span.date_time")
        info_el = li.select_one("p.link_dsc_txt:not(.dsc_sub)") or li.select_one("p.link_dsc_txt")
        gallery_el = li.select_one("p.dsc_sub a.sub_txt") or li.select_one(".dsc_sub a.sub_txt")
        if not link_el or not date_el:
            continue
        rows.append(
            Post(
                url=link_el.get("href"),
                date=date_el.get_text(strip=True),
                title=link_el.get_text(strip=True),
                snippet=info_el.get_text(" ", strip=True) if info_el else "",
                gallery=gallery_el.get_text(strip=True) if gallery_el else None,
            )
        )
    return rows


def wait_for_results(driver, timeout: int = 10) -> None:
    WebDriverWait(driver, timeout).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".sch_result_list li"))
    )


def clean_html_text(raw: str) -> str:
    return BeautifulSoup(raw, "lxml").get_text("\n", strip=True)


def build_mobile_url(post_url: str) -> Optional[str]:
    parsed = urlparse(post_url)
    if "dcinside.com" not in parsed.netloc or "view" not in parsed.path:
        return None
    qs = parse_qs(parsed.query)
    gall_id = qs.get("id", [None])[0]
    no = qs.get("no", [None])[0]
    if not gall_id or not no:
        return None
    return f"https://m.dcinside.com/board/{gall_id}/{no}"


def extract_comments_from_html(soup: BeautifulSoup) -> List[str]:
    comments = []
    for cont in soup.select(".all_comment_box .comment_wrap .comment_cont"):
        txt = cont.get_text(" ", strip=True)
        if txt:
            comments.append(txt)
    if comments:
        return comments
    for cont in soup.select(".comment_box .comment_txt"):
        txt = cont.get_text(" ", strip=True)
        if txt:
            comments.append(txt)
    return comments


def fetch_comments(
    session: requests.Session,
    url: str,
    soup: Optional[BeautifulSoup],
    config: CrawlConfig,
    article_date: Optional[str],
) -> List[str]:
    gall_id = None
    no = None
    gall_type = None
    board_type = ""
    e_s_n_o = None

    if soup:
        gall_id = (soup.find("input", {"name": "id"}) or {}).get("value")
        no = (soup.find("input", {"name": "no"}) or {}).get("value")
        gall_type = (soup.find("input", {"name": "_GALLTYPE_"}) or {}).get("value")
        e_s_n_o = (soup.find("input", {"name": "e_s_n_o"}) or {}).get("value")
        board_type_input = soup.find("input", {"name": "board_type"})
        if board_type_input and board_type_input.get("value"):
            board_type = board_type_input["value"]

    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if not gall_id:
        gall_id = qs.get("id", [None])[0]
    if not no:
        no = qs.get("no", [None])[0]
    if not gall_type:
        gall_type = "M"

    if gall_id and no and e_s_n_o and gall_type:
        cookie_token = session.cookies.get("ci_c", "")
        comments: Dict[str, Tuple[str, str]] = {}
        page = 1
        total_cnt = None

        while True:
            payload = {
                "id": gall_id,
                "no": no,
                "cmt_id": gall_id,
                "cmt_no": no,
                "focus_cno": "",
                "focus_pno": "",
                "e_s_n_o": e_s_n_o,
                "comment_page": page,
                "sort": config.comment_sort,
                "prevCnt": len(comments),
                "board_type": board_type,
                "_GALLTYPE_": gall_type,
                "secret_article_key": "",
                "ci_t": cookie_token,
            }
            resp = session.post(
                "https://gall.dcinside.com/board/comment/",
                data=payload,
                headers={
                    "User-Agent": config.user_agent,
                    "Referer": f"https://gall.dcinside.com/mgallery/board/view/?id={gall_id}&no={no}",
                    "Origin": "https://gall.dcinside.com",
                    "X-Requested-With": "XMLHttpRequest",
                },
                timeout=config.request_timeout,
            )
            if resp.status_code != 200:
                break
            try:
                data = resp.json()
            except ValueError:
                break

            if total_cnt is None:
                total_cnt = int(data.get("total_cnt", 0))

            for item in data.get("comments", []) or []:
                cno = item.get("no")
                reg_date = item.get("reg_date", "")
                memo = item.get("memo", "")
                if cno and cno not in comments:
                    comments[cno] = (reg_date, memo)

            if len(comments) >= (total_cnt or 0):
                break
            page += 1
            if not data.get("comments"):
                break
            time.sleep(0.1)

        def parse_comment_datetime(reg: str) -> datetime:
            reg = reg.strip()
            try:
                year = None
                if article_date and len(article_date) >= 4:
                    year = int(article_date.split(".")[0])
                if reg.count(":") == 1:
                    reg = f"{reg}:00"
                if year:
                    return datetime.strptime(f"{year}.{reg}", "%Y.%m.%d %H:%M:%S")
                return datetime.strptime(reg, "%m.%d %H:%M:%S")
            except Exception:
                return datetime.min

        comments_sorted = sorted(
            ((reg, clean_html_text(memo)) for reg, memo in comments.values()),
            key=lambda x: parse_comment_datetime(x[0]),
        )
        return [memo for _, memo in comments_sorted]

    if soup:
        return extract_comments_from_html(soup)
    return []


def fetch_post_detail(session: requests.Session, url: str, config: CrawlConfig) -> Tuple[Optional[str], List[str], Optional[str]]:
    headers = {"User-Agent": config.user_agent, "Referer": "https://search.dcinside.com"}
    content_text = None
    comments: List[str] = []
    date_value: Optional[str] = None

    def request_and_parse(target_url: str) -> Optional[BeautifulSoup]:
        resp = session.get(target_url, headers=headers, timeout=config.request_timeout)
        if resp.status_code != 200:
            return None
        return BeautifulSoup(resp.text, "lxml")

    soup = request_and_parse(url)
    if not soup:
        mobile_url = build_mobile_url(url)
        if mobile_url:
            soup = request_and_parse(mobile_url)

    if soup:
        content_el = soup.select_one(".write_div") or soup.select_one(".inbox")
        if content_el:
            content_text = content_el.get_text("\n", strip=True)
        date_el = soup.select_one(".gall_date")
        if date_el:
            date_value = date_el.get_text(strip=True)

    comments = fetch_comments(session=session, url=url, soup=soup, config=config, article_date=date_value)
    return content_text, comments, date_value
