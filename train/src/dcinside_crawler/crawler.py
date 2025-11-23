import json
import time
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from .config import CrawlConfig
from .http import build_driver, build_session
from .parsers import (
    fetch_post_detail,
    parse_search_items,
    wait_for_results,
)


def crawl(config: CrawlConfig) -> List[Dict]:
    config.validate()
    driver = build_driver(config.user_agent)
    session = build_session(config.user_agent)
    collected: List[Dict] = []
    page = config.start_page

    try:
        while len(collected) < config.limit:
            search_url = config.search_template.format(page=page)
            print(f"[page {page}] {search_url}", flush=True)
            driver.get(search_url)
            try:
                wait_for_results(driver, timeout=10)
            except Exception:
                print("Timed out waiting for search results; stopping.", flush=True)
                break

            posts = parse_search_items(driver.page_source)
            if not posts:
                print("No posts found on this page; stopping.", flush=True)
                break

            for post in tqdm(posts, desc=f"Page {page}", leave=False):
                content, comments, detail_date = fetch_post_detail(session, post.url, config)
                comments = [c for c in comments if isinstance(c, str) and c.strip() != ""]
                record = {
                    "date": detail_date or post.date,
                    "main": content or post.snippet or post.title,
                    "comments": comments,
                    "source_url": post.url,
                    "gallery": post.gallery or "",
                }
                collected.append(record)
                if len(collected) >= config.limit:
                    break
                time.sleep(config.sleep_between_posts)

            page += 1
            time.sleep(config.sleep_between_pages)
    finally:
        driver.quit()

    return collected


def write_output(data: List[Dict], config: CrawlConfig) -> None:
    Path(config.output_path).parent.mkdir(parents=True, exist_ok=True)
    if config.output_format == "ndjson":
        with open(config.output_path, "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        with open(config.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
