from __future__ import annotations

import datetime as dt
from typing import List
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from ..base_scraper import BaseScraper, ScrapeResult
from ..registry import register
from ...utils.io import shard_writer
from ...utils.logger import get_logger
from ...utils.schemas import RawDoc, Provenance

log = get_logger("usmle")

BASE_URL = "https://www.usmle.org/"
SAMPLE_QUESTION_PATHS = {
    "step1": "examinations/step-1/sample-questions",
    "step2": "examinations/step-2-ck/sample-questions",
    "step3": "examinations/step-3/sample-questions",
}


@register("usmle_official")
class USMLEOfficialScraper(BaseScraper):
    """Scrape publicly available USMLE sample questions for Steps 1–3."""

    name = "usmle_official"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def run(self) -> ScrapeResult:
        write, close = shard_writer(self.shards_dir, self.name, self.shard_max_records)
        total = 0
        async with await self._client() as client:
            for step, path in SAMPLE_QUESTION_PATHS.items():
                url = urljoin(BASE_URL, path)
                try:
                    html = await self._get_text(client, url)
                except Exception as exc:
                    log.error(f"[usmle] failed to fetch {url}: {exc}")
                    continue

                soup = BeautifulSoup(html, "html.parser")
                blocks = soup.select(".accordion__content")
                for idx, block in enumerate(blocks):
                    p = block.find("p")
                    answers = [li.get_text(strip=True) for li in block.select("li") if li.get_text(strip=True)]
                    question_text = p.get_text(strip=True) if p else ""
                    if not question_text or not answers:
                        continue

                    doc = RawDoc(
                        id=f"usmle:{step}:{total+1}",
                        title=f"USMLE {step.upper()} Sample Question",
                        text=question_text,
                        meta={
                            "type": "question",
                            "step": step,
                            "answers": answers,
                            "choice_count": len(answers),
                            "source_url": url,
                        },
                        prov=Provenance(
                            source="usmle_official",
                            source_url=url,
                            license="USMLE sample materials – educational use",
                            retrieved_at=dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        ),
                    )
                    write(doc.model_dump(mode="json"))
                    total += 1
                    if total >= self.max_docs:
                        break
                if total >= self.max_docs:
                    break

        close()
        log.info(f"[usmle] saved {total} sample questions")
        return ScrapeResult(total_fetched=total, shards_path=self.shards_dir)
