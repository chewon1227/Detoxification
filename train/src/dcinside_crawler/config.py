from dataclasses import dataclass


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
)


@dataclass
class CrawlConfig:
    search_template: str = "https://search.dcinside.com/post/p/{page}/sort/latest/q/.EB.82.9C.EB.AF.BC"
    limit: int = 1000
    start_page: int = 1
    output_path: str = "dcinside_data.json"
    output_format: str = "json"  # json | ndjson
    user_agent: str = DEFAULT_USER_AGENT
    request_timeout: int = 15
    sleep_between_posts: float = 0.2
    sleep_between_pages: float = 0.5
    comment_sort: str = "D"  # D = 등록순, N = 최신순

    def validate(self) -> None:
        if self.output_format not in {"json", "ndjson"}:
            raise ValueError("output_format must be 'json' or 'ndjson'")
