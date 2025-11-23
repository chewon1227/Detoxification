import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from dcinside_crawler.config import CrawlConfig  # noqa: E402
from dcinside_crawler.crawler import crawl, write_output  # noqa: E402


def parse_args() -> CrawlConfig:
    parser = argparse.ArgumentParser(description="Crawl dcinside search results.")
    parser.add_argument("--limit", type=int, default=1000, help="Number of rows to collect.")
    parser.add_argument("--start-page", type=int, default=1, help="Search page to start from.")
    parser.add_argument(
        "--search-template",
        default="https://search.dcinside.com/post/p/{page}/sort/latest/q/.EB.82.9C.EB.AF.BC",
        help="Search URL template; must include {page}.",
    )
    parser.add_argument("--output", default="dcinside_data.json", help="Output file path.")
    parser.add_argument("--output-format", default="json", choices=["json", "ndjson"], help="Output file format.")
    parser.add_argument("--comment-sort", default="D", choices=["D", "N"], help="Comment sort: D=등록순, N=최신순.")
    parser.add_argument("--user-agent", default=None, help="Override User-Agent header.")
    parser.add_argument("--request-timeout", type=int, default=15, help="HTTP request timeout seconds.")
    args = parser.parse_args()

    return CrawlConfig(
        search_template=args.search_template,
        limit=args.limit,
        start_page=args.start_page,
        output_path=args.output,
        output_format=args.output_format,
        comment_sort=args.comment_sort,
        user_agent=args.user_agent or CrawlConfig().user_agent,
        request_timeout=args.request_timeout,
    )


def main() -> None:
    config = parse_args()
    data = crawl(config)
    write_output(data, config)
    print(f"Wrote {len(data)} rows to {config.output_path}")


if __name__ == "__main__":
    main()
