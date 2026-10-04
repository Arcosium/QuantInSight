"""
NPS Swarm v1.0 - Naver Real-time Search Tool
Uses Playwright for headless browser scraping of Naver Finance & News.
"""
import asyncio
from playwright.async_api import async_playwright


async def naver_realtime_search(query: str, search_type: str = "news", max_results: int = 10) -> str:
    """
    Scrape Naver for real-time Korean market information using Playwright.

    Args:
        query: Search keyword (e.g., "삼성전자", "미국 금리")
        search_type: "news" for Naver News, "finance" for Naver Finance
        max_results: Maximum number of results

    Returns:
        Formatted string of search results with titles, snippets, and dates
    """
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            page = await context.new_page()

            if search_type == "news":
                results = await _scrape_naver_news(page, query, max_results)
            elif search_type == "finance":
                results = await _scrape_naver_finance(page, query, max_results)
            else:
                results = await _scrape_naver_news(page, query, max_results)

            await browser.close()
            return results
    except Exception as e:
        return f"[Naver Search Error] {str(e)}"


async def _scrape_naver_news(page, query: str, max_results: int) -> str:
    """Scrape Naver News search results."""
    url = f"https://search.naver.com/search.naver?where=news&query={query}&sort=1"
    await page.goto(url, wait_until="domcontentloaded", timeout=15000)
    await asyncio.sleep(1)

    articles = []
    news_items = await page.query_selector_all("div.news_area")

    for item in news_items[:max_results]:
        try:
            title_el = await item.query_selector("a.news_tit")
            title = await title_el.get_attribute("title") if title_el else ""
            link = await title_el.get_attribute("href") if title_el else ""

            desc_el = await item.query_selector("div.news_dsc")
            desc = await desc_el.inner_text() if desc_el else ""

            info_el = await item.query_selector("div.info_group span.info")
            date = await info_el.inner_text() if info_el else ""

            articles.append(f"📰 {title}\n   날짜: {date}\n   요약: {desc[:150]}\n   링크: {link}")
        except Exception:
            continue

    if not articles:
        return f"[Naver News] '{query}'에 대한 뉴스를 찾지 못했습니다."

    header = f"[Naver News 실시간 검색] 쿼리: '{query}' | {len(articles)}건\n"
    return header + "\n\n".join(articles)


async def _scrape_naver_finance(page, query: str, max_results: int) -> str:
    """Scrape Naver Finance stock information."""
    url = f"https://finance.naver.com/search/searchList.naver?query={query}"
    await page.goto(url, wait_until="domcontentloaded", timeout=15000)
    await asyncio.sleep(1)

    stocks = []
    rows = await page.query_selector_all("table.tbl_search tbody tr")

    for row in rows[:max_results]:
        try:
            name_el = await row.query_selector("td a")
            name = await name_el.inner_text() if name_el else ""
            href = await name_el.get_attribute("href") if name_el else ""

            code = ""
            if "code=" in (href or ""):
                code = href.split("code=")[-1]

            tds = await row.query_selector_all("td")
            price = await tds[1].inner_text() if len(tds) > 1 else ""
            change = await tds[2].inner_text() if len(tds) > 2 else ""

            stocks.append(f"📊 {name} ({code}) | 현재가: {price} | 변동: {change}")
        except Exception:
            continue

    if not stocks:
        return f"[Naver Finance] '{query}'에 대한 종목을 찾지 못했습니다."

    header = f"[Naver Finance 검색] 쿼리: '{query}' | {len(stocks)}건\n"
    return header + "\n".join(stocks)
