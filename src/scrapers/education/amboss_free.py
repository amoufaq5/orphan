from __future__ import annotations

import datetime as dt
from typing import List

from bs4 import BeautifulSoup

from ..base_scraper import BaseScraper, ScrapeResult
from ..registry import register
from ...utils.io import shard_writer
from ...utils.logger import get_logger
from ...utils.schemas import RawDoc, Provenance

log = get_logger("amboss")

AMBOSS_FREE_URLS: List[str] = [
    "https://www.amboss.com/us/student-blog/usmle-step-1-practice-questions",
    "https://www.amboss.com/us/student-circle/usmle-step-2-ck-practice-questions",
    "https://www.amboss.com/us/student-circle/usmle-step-3-practice-questions",
]


@register("amboss_free")
class AmbossFreeScraper(BaseScraper):
    """Scrape publicly available AMBOSS practice question articles."""

    name = "amboss_free"

    async def run(self) -> ScrapeResult:
        write, close = shard_writer(self.shards_dir, self.name, self.shard_max_records)
        total = 0
        async with await self._client() as client:
            for url in AMBOSS_FREE_URLS:
                if total >= self.max_docs:
                    break
                try:
                    html = await self._get_text(client, url)
                except Exception as exc:
                    log.error(f"[amboss] failed to fetch {url}: {exc}")
                    continue

                soup = BeautifulSoup(html, "html.parser")
                article = soup.find("article") or soup
                blocks = article.select("h2, h3, p, li")
                if not blocks:
                    continue

                text_parts: List[str] = []
                for node in blocks:
                    content = node.get_text(" ", strip=True)
                    if content:
                        text_parts.append(content)

                if not text_parts:
                    continue

                doc = RawDoc(
                    id=f"amboss:{total+1}",
                    title=soup.find("title").get_text(strip=True) if soup.find("title") else "AMBOSS Practice Questions",
                    text="\n".join(text_parts),
                    meta={
                        "type": "education_article",
                        "source_url": url,
                    },
                    prov=Provenance(
                        source="amboss_free",
                        source_url=url,
                        license="AMBOSS public educational content",
                        retrieved_at=dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    ),
                )
                write(doc.model_dump(mode="json"))
                total += 1
                if total >= self.max_docs:
                    break

        close()
        log.info(f"[amboss] saved {total} articles")
        return ScrapeResult(total_fetched=total, shards_path=self.shards_dir)
