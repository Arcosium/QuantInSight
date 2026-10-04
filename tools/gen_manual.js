// ArQuant v1.0 사용설명서 생성기 — 현재 시스템(코드/UI) 기준 전면 재작성 (2026-05-24)
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, ExternalHyperlink,
  TabStopType, TabStopPosition, TableOfContents, HeadingLevel,
  BorderStyle, WidthType, ShadingType, PageNumber, PageBreak,
} = require("docx");

const FONT = "Malgun Gothic"; // 한글 친화 폰트 (Word가 미설치 시 CJK 폰트로 자동 대체)
const CONTENT_W = 9360;       // US Letter, 1" 여백

// ── helpers ───────────────────────────────────────────────
const P = (text, opts = {}) =>
  new Paragraph({ spacing: { after: 120, line: 312 }, ...opts,
    children: opts.children || [new TextRun({ text, ...(opts.run || {}) })] });

const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });

const bullet = (text, level = 0) =>
  new Paragraph({ numbering: { reference: "bullets", level },
    spacing: { after: 80, line: 300 }, children: [new TextRun(text)] });

const num = (text) =>
  new Paragraph({ numbering: { reference: "steps", level: 0 },
    spacing: { after: 80, line: 300 }, children: [new TextRun(text)] });

const note = (label, text) =>
  new Paragraph({
    spacing: { before: 80, after: 160, line: 300 },
    shading: { fill: "FFF6E0", type: ShadingType.CLEAR },
    border: { left: { style: BorderStyle.SINGLE, size: 18, color: "E0A300", space: 8 } },
    children: [new TextRun({ text: `${label}  `, bold: true }), new TextRun(text)],
  });

const border = { style: BorderStyle.SINGLE, size: 1, color: "C9D3DF" };
const borders = { top: border, bottom: border, left: border, right: border,
  insideHorizontal: border, insideVertical: border };

function table(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (txt, w, opts = {}) => new TableCell({
    borders,
    width: { size: w, type: WidthType.DXA },
    margins: { top: 60, bottom: 60, left: 110, right: 110 },
    shading: opts.head ? { fill: "1F3B57", type: ShadingType.CLEAR } : (opts.shade ? { fill: "EEF3F8", type: ShadingType.CLEAR } : undefined),
    children: [new Paragraph({ spacing: { after: 0, line: 276 },
      children: [new TextRun({ text: String(txt), bold: !!opts.head,
        color: opts.head ? "FFFFFF" : undefined, size: 19 })] })],
  });
  const trs = [new TableRow({ tableHeader: true,
    children: headers.map((h, i) => cell(h, widths[i], { head: true })) })];
  rows.forEach((r, ri) => trs.push(new TableRow({
    children: r.map((c, i) => cell(c, widths[i], { shade: ri % 2 === 1 })) })));
  return new Table({ width: { size: total, type: WidthType.DXA },
    columnWidths: widths, rows: trs });
}

const link = (text, url) => new ExternalHyperlink({
  link: url, children: [new TextRun({ text, style: "Hyperlink" })] });

// ── document ──────────────────────────────────────────────
const children = [];

// 표지
children.push(
  new Paragraph({ spacing: { before: 2600, after: 200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "ArQuant v1.0", bold: true, size: 64, color: "1F3B57" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: [new TextRun({ text: "AI 멀티에셋 퀀트 트레이딩 시스템", size: 30, color: "44607A" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
    children: [new TextRun({ text: "사용설명서", bold: true, size: 40 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
    children: [link("https://arquant.ai-ve.uk", "https://arquant.ai-ve.uk")] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
    children: [new TextRun({ text: "최종 업데이트: 2026-06-11", size: 20, color: "8A99AB" })] }),
  new Paragraph({ children: [new PageBreak()] }),
);

// 목차
children.push(H1("목차"),
  new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ children: [new PageBreak()] }));

// 1. 소개
children.push(H1("1. ArQuant 소개"));
children.push(P("ArQuant는 10명의 AI 에이전트(+뉴스 분류·큐레이터)가 협업하여 글로벌 지수·실시간 시황·3년치 일봉/수급·분봉·DART 공시·증권 속보를 종합 분석하고, 국내주식·해외주식에 더해 채권·원자재 ETF까지 통합 관리하며 한국투자증권(KIS) OpenAPI로 실거래를 수행하는 자동매매 시스템입니다. 정규장(KRX)뿐 아니라 NXT 대체거래소 시간외(프리마켓 08:00–08:50 / 애프터마켓 15:50–20:00)와 미국 야간 세션(22:30–05:00)까지 거래합니다."));
children.push(P("사용자는 웹 브라우저 또는 모바일 앱으로 접속해 ‘실행’만 누르면, 시스템이 장 개장·정기 주기에 맞춰 스스로 분석하고 매수/매도합니다. 사용자는 전략을 고르거나, 채팅으로 AI에게 직접 지시하거나, 수익률을 지켜보기만 하면 됩니다."));
children.push(H2("1.1 무엇을 자동으로 하나요"));
[
  "장이 열려 있는 동안 매시 정각(:00) + 개장 순간에 분석 사이클을 자동 수행합니다(2패스 종목 선정).",
  "글로벌리서치팀장의 4분할 자산배분(주식/채권/원자재/현금)에 따라 채권·원자재 ETF 비중도 자동 운용합니다.",
  "뉴스를 약 10~15분마다 수집하고 한국/미국/공통으로 자동 분류합니다.",
  "리스크 게이트(결정론 규칙) + DART 공시·재무 재심을 통과한 종목만 매수합니다.",
  "보유 종목은 사후관리실장이 매 사이클 매도/유지/익절·손절을 재결정합니다(매수 시 기록한 thesis 권고 참고).",
  "수익률·자산 평가는 증권사 집계값이 아닌 자체 '실거래 원장' 기준으로 계산해 결제 지연·전산 글리치에 흔들리지 않습니다.",
  "모든 의사결정·체결 결과를 한국어로 보고하고 화면·모바일 알림으로 전달합니다.",
].forEach((t) => children.push(bullet(t)));
children.push(note("핵심", "사용자가 매번 종목을 고를 필요가 없습니다. 전략 설정값과 위험 한도를 정해두면 나머지는 AI 팀이 처리합니다. 기본 설정값으로 바로 실거래로 동작합니다."));

// 2. 준비물
children.push(H1("2. 시작하기 전에 — 준비물"));
children.push(P("회원가입 시 로컬 LLM와 KIS 키·계좌번호가 필요합니다. 관전 모드는 아이디와 비밀번호만으로 가입할 수 있습니다."));
children.push(table(
  ["항목", "필수", "용도"],
  [
    ["로컬 LLM", "필수", "모든 AI 분석 호출에 사용 (비용 발생)"],
    ["KIS App Key / Secret", "거래 시 필수", "한국투자증권 시세 조회·주문"],
    ["KIS 계좌번호", "거래 시 필수", "실거래가 일어나는 계좌"],
    ["거래 환경", "거래 시 필수", "실전투자 / 모의투자 중 선택 (Base URL 자동 설정)"],
  ],
  [2600, 1400, 5360]));
children.push(note("비용 안내", "AI 분석은 로컬 LLM 서버 사용량에 따라 과금됩니다. 화면 우상단에서 추정 비용을 확인할 수 있습니다."));
children.push(P("아래 2.1~2.2는 위 키를 처음 발급받는 분을 위한 단계별 안내입니다. 발급한 키는 3장(회원가입) 화면에 그대로 붙여넣으면 됩니다. (각 사이트의 메뉴 이름은 바뀔 수 있으니, 굵게 표시한 동작 흐름과 주소를 기준으로 따라오세요.)"));

children.push(H2("2.1 로컬 LLM 발급 (필수)"));
children.push(P("로컬 LLM 서버 키는 ArQuant의 모든 AI 분석과 Hermes 검색 리서치에 사용됩니다."));
children.push(P("1) 웹브라우저에서 아래 주소에 접속합니다.", { run: { bold: true } }));
children.push(P(null, { children: [new TextRun("    → "), link("http://127.0.0.1:8080/v1", "http://127.0.0.1:8080/v1")] }));
children.push(P("2) 우측 상단 ‘Sign in’을 눌러 Google·GitHub 또는 이메일로 가입/로그인합니다."));
children.push(P("3) 로그인 후 키 발급 페이지로 이동합니다."));
children.push(P("로컬 LLM 서버에서 API Key를 생성합니다."));
children.push(P("4) ‘Create Key’(키 만들기)를 누르고 이름을 자유롭게(예: arquant) 입력한 뒤 생성합니다."));
children.push(P("5) 화면에 표시된 키(sk- 로 시작)를 즉시 복사합니다. 이 키는 다시 볼 수 없으니 안전한 곳에 보관하세요."));
children.push(P("6) 크레딧(사용 금액)을 충전합니다. 충전이 없으면 분석이 멈춥니다."));
children.push(P("로컬 LLM 서버의 Billing 메뉴에서 사용량과 잔액을 관리합니다."));
children.push(note("확인", "올바른 키는 sk- 로 시작합니다. 크레딧이 0이 되면 AI 분석이 중단되며, 화면의 💵 배지를 누르면 잔액 확인·충전 페이지로 연결됩니다(13장)."));

children.push(H2("2.2 한국투자증권(KIS) App Key·Secret 발급 (실거래 필수)"));
children.push(P("실제(또는 모의) 주문·시세 조회에 사용합니다. 먼저 한국투자증권 계좌와 로그인 ID가 있어야 합니다. 계좌가 없다면 ‘한국투자증권’ 모바일 앱에서 비대면으로 개설할 수 있습니다."));
children.push(P("1) 한국투자증권 KIS Developers(개발자센터) 포털에 접속해 한국투자증권 ID로 로그인합니다.", { run: { bold: true } }));
children.push(P(null, { children: [new TextRun("    → "), link("https://apiportal.koreainvestment.com", "https://apiportal.koreainvestment.com")] }));
children.push(P("2) ‘API 신청/이용신청’ 메뉴에서 약관에 동의하고 OpenAPI 사용을 신청합니다."));
children.push(P("3) ‘모의투자’ 또는 ‘실전투자’ 중 사용할 유형을 선택해 신청합니다. 처음에는 가짜 돈으로 안전하게 검증할 수 있는 ‘모의투자’를 권장합니다."));
children.push(P("4) 신청이 완료되면 ‘My Page(앱 관리/이용현황)’에서 App Key와 App Secret을 확인하고 둘 다 복사합니다."));
children.push(P("5) 거래할 한국투자증권 계좌번호를 준비합니다. 보통 ‘앞 8자리 + 상품 2자리 = 10자리’ 형태입니다(예: 12345678-01). 회원가입 화면에는 예시 형식 그대로 넣으면 됩니다."));
children.push(note("매우 중요", "모의투자 키와 실전투자 키는 서로 다릅니다. 회원가입 시 ‘거래 환경’을 키 종류와 똑같이(실전/모의) 골라야 합니다. Base URL은 거래 환경 선택에 따라 자동 설정되므로 직접 입력할 필요가 없습니다. 모의 키로 실전 환경을 고르면 로그인은 되는데 주문이 계속 실패합니다. App Secret은 비밀번호와 같으니 절대 타인과 공유하지 마세요(계정 복구의 본인확인에도 쓰입니다)."));

// 3. 회원가입·로그인·복구
children.push(H1("3. 회원가입 · 로그인 · 계정 복구"));
children.push(P("처음 접속하면 로그인 화면이 뜹니다. 상단에 ‘로그인 / 최초 등록’ 탭이 있습니다. 비밀번호는 복호화 불가능한 해시(argon2id)로만 저장됩니다."));
children.push(H2("3.1 최초 등록"));
children.push(P("‘최초 등록’ 탭에서 아래 항목을 입력하고 ‘등록하고 시작’을 누릅니다."));
[
  "아이디 — 3자 이상. 입력하면 실시간으로 중복 여부를 확인해 줍니다(✓ 사용 가능 / ✗ 이미 사용 중).",
  "비밀번호 — 10자 이상 + 특수문자 1개 이상(실시간 충족 여부 표시).",
  "로컬 LLM — 필수.",
  "한국투자증권 App Key / App Secret — 거래 시 필요(선택 입력 가능).",
  "한국투자증권 계좌번호 — 예: 12345678-01.",
  "거래 환경 — ‘실전투자(실거래)’ 또는 ‘모의투자(페이퍼)’ 중 선택(Base URL 자동 설정).",
  "이 기기에서 7일간 로그인 유지 — 체크하면 7일간 자동 로그인.",
].forEach((t) => children.push(bullet(t)));
children.push(note("DART는 입력 불필요", "예전과 달리 DART(공시) 키는 사용자가 입력하지 않습니다. 공시·재무 재심은 시스템이 자체 키로 자동 수행합니다."));
children.push(H2("3.2 로그인"));
children.push(P("등록한 아이디·비밀번호로 로그인합니다. ‘7일간 로그인 유지’를 체크하면 다시 로그인할 필요가 줄어듭니다. 로그인·등록·복구 시도에는 횟수 제한과 감사 기록이 적용됩니다."));
children.push(H2("3.3 아이디·비밀번호 찾기 (계정 복구)"));
children.push(P("로그인 화면의 ‘아이디/비밀번호를 잊으셨나요?’를 누르면 복구 패널이 열립니다. 이메일·문자 인증은 없으며, 가입 때 입력한 한국투자증권 정보로 본인을 확인합니다."));
[
  "아이디 찾기 — ‘계좌번호 + App Secret’을 입력하면 등록된 아이디를 알려줍니다.",
  "비밀번호 재설정 — ‘아이디 + 계좌번호 + App Secret + 새 비밀번호’를 입력하면 즉시 재설정됩니다.",
].forEach((t) => children.push(bullet(t)));
children.push(note("보관 필수", "계좌번호와 App Secret이 곧 복구 열쇠입니다. 이 두 값을 바꾸면 이후 복구 시 새 값으로 입력해야 하니 꼭 기억해 두세요."));

// 4. 화면 한눈에 보기
children.push(H1("4. 화면 한눈에 보기"));
children.push(P("상단바 좌측에 로고, 우측에 💵 API 비용 배지 · 👤 프로필 버튼(내 계정 관리, 15장)이 있습니다. 실행 상태 배지는 ‘에이전트 통신 로그’ 카드 헤더에 표시되고, ▶ 실행 / ⏹ 중지 / 지시 입력창 / ➤ 전송은 화면 하단의 컨트롤 바에 있습니다. 본문은 탭으로 구성됩니다."));
children.push(table(
  ["탭", "역할"],
  [
    ["📊 대시보드", "세션·시간·뉴스·사이클·실매매·장 상태 + AI 통신 로그 + (PC) 에이전트 목록"],
    ["💰 수익률", "누적수익 KPI 8종 + 누적수익 곡선(벤치마크 겹쳐보기)"],
    ["💼 보유종목", "예수금·총평가·보유 종목 + 전체 거래 내역(상세·FIFO)"],
    ["📰 뉴스", "수집된 증권 속보 (한국/미국/공통 필터)"],
    ["⚙️ 전략", "현재 전략 설정값 확인·편집 + 운용지원 ON/OFF"],
    ["🛡️ ADMIN", "관리자 전용 — 전역 설정·회원 관리·피드백 답글 (관리자에게만 노출)"],
  ],
  [2000, 7360]));
children.push(note("프로필 버튼의 빨간 점", "👤 버튼이나 ‘피드백/버그 제보’ 옆에 빨간 점이 보이면, 보낸 피드백에 관리자(사장님) 답글이 달렸다는 뜻입니다(15.6 참고)."));

// 5. 실행과 중지
children.push(H1("5. 실행 · 중지와 거래 세션"));
children.push(num("화면 하단 ▶ 실행 을 누르면 무한 시장 감시 루프가 시작됩니다."));
children.push(num("이후 1시간 주기 + 한국/미국 장 개장 순간에 분석 사이클이 자동 수행됩니다."));
children.push(num("멈추려면 ⏹ 중지 를 누릅니다."));
children.push(P("재접속하면 실행 상태가 서버와 자동 동기화됩니다. 서버 재시작 후에는 감시 루프가 멈춰 있을 수 있으니 필요 시 다시 ▶ 실행 을 눌러주세요."));
children.push(H3("거래 트리거 (3가지)"));
[
  "▶ 실행 직후 첫 회 — 장중이면 누적 뉴스로 즉시 1회",
  "장 개장 순간 — 세션 진입 시 1회",
  "정기 — 장이 열려 있는 동안 매시 정각(:00)에 1회 (서버를 재시작해도 정각 기준은 변하지 않습니다)",
].forEach((t) => children.push(bullet(t)));
children.push(P("단, 다음 경우엔 사이클을 건너뜁니다: 직전 사이클이 5분 이내(중복 방지), 휴장일(아래 참고), 가용 예수금이 5,000원 미만(최저가 1주도 못 사므로 분석 비용만 듦)."));
children.push(H3("거래 세션 (KST 기준)"));
children.push(table(
  ["세션", "시간", "동작"],
  [
    ["KR_PRE_MARKET", "08:00–08:50", "NXT(대체거래소) 시간외 프리마켓 — 지정가 거래"],
    ["KR_TRADING", "09:00–15:30", "국내 정규장(KRX) 거래·정기 사이클"],
    ["KR_CLOSE_REVIEW", "15:30–16:30", "장 마감 리뷰 (매매 없음)"],
    ["KR_AFTER_MARKET", "15:50–20:00", "NXT 시간외 애프터마켓 — 지정가 거래"],
    ["US_TRADING", "22:30–05:00", "미국 야간 거래·개장 트리거"],
    ["OFF_HOURS", "그 외", "뉴스만 수집, 거래 없음"],
  ],
  [2500, 2000, 4860]));
children.push(note("NXT 시간외 매매", "프리/애프터마켓은 NXT 대체거래소에서 지정가(현재가 ±0.5% 밴드)로 거래합니다. 전략 탭에서 시간외 매매 전체·프리·애프터를 각각 켜고 끌 수 있습니다. NXT에 상장되지 않아 거래가 거부된 종목은 시스템이 자동으로 기억해(전 계정 공유) 이후 시간외 시간에는 시도하지 않고 정규장에서만 거래합니다. 모의투자 계정은 NXT를 지원하지 않아 시간외 매매가 비활성화됩니다."));
children.push(note("휴장일 자동 확인", "주말은 당연히 거래하지 않습니다. 그 외 평일에는 개장 5분 후 한국투자증권 실시세(코스피 지수·대표 종목 당일 시세)로 ‘오늘 실제로 장이 열렸는지’를 한 번 확인합니다. 실제로 열렸으면 내장 휴장일 목록과 무관하게 정상 거래하고, 당일 시세가 없으면(미등록 임시휴장 포함) 그날은 자동으로 사이클을 건너뜁니다. 확인이 불가하면 기존 휴장일 목록으로 안전하게 판단합니다."));

// 6. 대시보드 탭
children.push(H1("6. 대시보드 탭"));
children.push(P("시스템의 현재 상태를 한눈에 보여줍니다. 상단 상태 카드(8종): 세션 · KST 시간 · 감지된 뉴스 수 · 다음 사이클까지 남은 시간 · 완료 사이클 수 · 실매매 체결 수 · 활성 전략 · 장 상태(🟢 장중 / ⚪ 장외). 모바일에서는 ‘전체 상태 보기’로 펼쳐 볼 수 있습니다."));
children.push(P("‘💬 에이전트 통신 로그’에는 각 AI의 분석·결정이 한국어로 실시간 표시됩니다(마크다운 표도 표로 렌더). 헤더의 실행 상태 배지로 현재 동작을 알 수 있고, ‘초기화’ 버튼으로 표시 로그를 비울 수 있습니다."));
children.push(P("PC 화면 왼쪽 ‘🏛️ 에이전트’ 목록의 이름을 누르면 하단 입력창에 ‘@이름 ’이 자동 입력되어 그 AI에게 바로 지시할 수 있습니다(11장). 모바일에서는 사이드바 대신 입력창의 @ 자동완성으로 호출합니다."));

// 7. 수익률 탭
children.push(H1("7. 수익률 탭"));
children.push(P("운용 성과를 보여줍니다. 위쪽에 KPI 카드 8종이 있고, 아래에 누적수익 곡선이 있습니다."));
children.push(H2("7.1 KPI 카드 (8종)"));
children.push(table(
  ["카드", "의미"],
  [
    ["평가금액 변동 (전체)", "운용 시작 이후 평가금액 변동(원/%) — 미실현 손익 포함"],
    ["오늘 / 이번주 / 이번달", "각 기간 시작 대비 평가금액 변동(원/%)"],
    ["최대낙폭 (MDD)", "고점 대비 최대 하락폭(%)"],
    ["승률", "매도 거래 중 이익으로 끝난 비율 (N승 / M매도)"],
    ["누적 수익 (실현)", "실제 체결된 매도의 실현손익 합계 — 거래비용 반영(해외 매수·매도 각 0.3%)"],
    ["현재 평가액", "현재 자산 평가금액 (모의투자 계정은 숨김)"],
  ],
  [2600, 6760]));
children.push(note("실거래 원장 기준 (중요)", "평가금액·수익률은 증권사 '계좌 집계값'이 아니라 ArQuant가 직접 굴리는 실거래 원장으로 계산합니다. 증권사 집계값은 해외 매도대금의 결제 지연(T+2)·전산 글리치로 일시적으로 크게 틀어질 수 있어서, 시스템은 ① 최초 1회 보유·예수금(미결제 매도대금 포함)으로 원장을 만들고 ② 이후엔 실제 체결만으로 현금·보유를 굴리며 ③ 시세·환율로 평가합니다. 입출금이나 수동 매매를 했다면 원장과 실계좌가 어긋나므로 재동기화가 필요합니다(괴리가 생기면 30분 내 경고 알림이 옵니다)."));
children.push(H2("7.2 누적수익 곡선"));
children.push(P("차트는 ‘평가금액’이 아니라 시작점을 0원으로 둔 누적수익(원)의 변화를 보여줍니다. 상단 버튼으로 보기를 전환합니다."));
children.push(table(
  ["보기", "정의"],
  [
    ["실시간", "장 운영시간(한국 09:00–15:30 / 미국 22:30–05:00) 포인트만 5분 단위로 표시"],
    ["일별", "날짜별 마지막 값"],
    ["월별", "월별 마지막 값"],
  ],
  [1600, 7760]));
children.push(P("‘📊 벤치마크 ON’을 누르면 코스피·나스닥 지수를 시작값에 맞춰 내 곡선에 점선으로 겹쳐 비교할 수 있습니다(범례 표시). 가로축 시각은 모두 KST 기준입니다."));
children.push(note("장 운영시간 외 동결", "장 시간이 아니면 가격이 움직이지 않으므로 평가금액 포인트를 기록하지 않고, 수익률 탭도 자동 갱신을 멈춰 마지막 장중 값으로 동결됩니다. 그래서 장외에는 ‘실시간 수익률’이 변하지 않습니다."));

// 8. 보유종목 탭
children.push(H1("8. 보유종목 탭"));
children.push(H2("8.1 보유 종목 / 잔고"));
children.push(P("예수금·총평가·평가손익과 보유 종목 목록을 카테고리 배지(국내주식/해외주식)와 함께 보여줍니다. 미국 종목은 달러 가격과 원화 환산액(≈N원)을 함께 표시합니다. 자동 폴링은 10분 주기이며 장외(OFF_HOURS)에는 일시정지됩니다. 🔄 새로고침 버튼으로 즉시 강제 갱신할 수 있습니다."));
children.push(H2("8.2 전체 거래 내역"));
children.push(P("실제로 체결된 매수/매도 내역이 KST 시각과 함께 나열됩니다. 한 행을 누르면 상세 정보가 그 자리에서 펼쳐집니다."));
[
  "매수: 정확한 체결가(주문 전후 보유 평균단가 차이로 역산) 또는 추정 체결가 + 총 매입액",
  "매도: 추정 매도가 + 매도 시점 평가손익",
  "FIFO 매칭표: 어떤 매수분이 어떤 매도와 짝지어졌는지 + 로트별 실현 손익",
  "💰 실현 손익 합계 (🟢 이익 / 🔴 손실 색상)",
].forEach((t) => children.push(bullet(t)));
children.push(P("🗑️ 비우기 버튼은 거래 내역 표시만 초기화하며 시스템 로그에는 영향을 주지 않습니다."));
children.push(note("비우기 후에도 통계 유지", "거래 내역을 비워도 승률·매도 횟수 등 누적 통계는 사라지지 않습니다. 비우기 직전의 실현손익 통계를 따로 적립해 이후 거래에 합산하므로, 거래 목록을 비워도 승률 표시가 유지됩니다. 누적 평가금액(수익) 곡선도 별도로 보존되어 영향받지 않습니다."));

// 9. 뉴스 탭
children.push(H1("9. 뉴스 탭"));
children.push(P("증권 속보를 약 15분마다 수집해, 한국(🇰🇷)/미국(🇺🇸)/공통(🌐)으로 자동 분류하여 보여줍니다. 상단 필터(전체/국내/미국/공통)로 걸러 볼 수 있고, 각 헤드라인에는 마켓 배지와 크롤 시각이 표시됩니다. 제목에 원문 링크가 있으면 눌러서 새 탭으로 열 수 있습니다. 누적 헤드라인이 많으면 큐레이터가 굵직한 40건만 선별해 AI에 전달합니다."));
children.push(P("‘🗑️ 로그 정리’ 버튼은 뉴스 로그를 정리하되 최근 20건은 유지합니다."));

// 10. 전략 탭
children.push(H1("10. 전략 탭"));
children.push(P("'전략' 탭에서 현재 적용 중인 모든 파라미터(53개)를 그룹별로 보고, 한국어 라벨·단위(%, 배, 일, 건)로 직접 수정해 '변경값 적용'을 누르면 재시작 없이 즉시 라이브 반영됩니다. 예전의 '방어형/균형형/공격형' 프리셋은 폐지되었고, 균형형 기준의 단일 기본값에서 출발해 내 계정에 맞게 직접 조정하는 방식입니다."));
children.push(table(
  ["그룹", "대표 항목 (기본값)"],
  [
    ["사이징", "1주문 예수금 비율 10% · 사이클 누적 예산 25% · 예수금 안전마진 +10%"],
    ["리스크", "신규매수 차단 손익 -5% · 단일 종목 비중 한도 15% · 사이클당 최대 매매 2건"],
    ["종목 필터", "최소 퀀트점수 6점 · RSI 과매수 스킵 · 외국인 순매수 요구 등"],
    ["퀀트 지표 가중치", "RSI/MACD/ADX/VWAP/거래량/모멘텀/CMF/수급/52주 — 가중치 직접 조정(음수 허용)"],
    ["점수 차원", "퀀트 60 : 뉴스 25 : 매크로 15 (결정론 점수 엔진)"],
    ["레짐 대응", "자동 익절 12% · 자동 손절 5% · 데이트레이딩 ON · 매크로 매수 게이트"],
    ["NXT 시간외", "시간외 매매 전체/프리/애프터 ON·OFF · 지정가 슬리피지 밴드 0.5%"],
    ["채권 ETF", "채권 슬리브 ON · 비중 상한 40% · 리밸런싱 데드존 ±3%"],
    ["원자재 ETF", "원자재 슬리브 ON · 비중 상한 20% · 리밸런싱 데드존 ±3%"],
    ["제도권 파이프라인", "퀀트랭크 매수 종목 수 · 리스크 기반 사이징 · 유니버스 스크리닝"],
  ],
  [2400, 6960]));
children.push(H2("10.1 전략 커스터마이즈"));
children.push(P("각 항목은 현재 적용값이 입력칸에 그대로 표시됩니다. 수정 후 '변경값 적용'으로 라이브 오버라이드하거나, '되돌리기'로 현재 적용값을 다시 불러올 수 있습니다. 미국주식 허용 등 일부 정책 항목은 운용지원실장이 자동 조정할 수 없고 사용자만 바꿀 수 있습니다."));
children.push(P("‘현재 적용’ 카드에는 지금 적용 중인 전략, 운용지원실장이 내 파라미터를 마지막으로 조정한 시각, 그리고 현재 적용된 모든 상세 설정이 함께 표시됩니다."));
children.push(H2("10.2 운용지원 ON / OFF"));
children.push(P("‘적용 가능 전략’ 제목 오른쪽의 ‘🛠 운용지원 ON/OFF’ 토글로, 운용지원실장이 내 계정의 전략 파라미터를 진단·조정하도록 켜거나 끌 수 있습니다(계정별 독립, 기본 ON). 끄면 자동 조정을 멈춥니다."));

// 11. @멘션
children.push(H1("11. AI에게 직접 지시하기 (@멘션)"));
children.push(P("화면 하단 입력창에 `@에이전트명 지시내용` 형식으로 입력하면 해당 AI에게 즉시 지시할 수 있습니다. `@`를 생략하면 주식운용실장에게 자동 전달됩니다(@ 입력 시 이름 자동완성)."));
[
  "@주식운용실장 미국 기술주 비중 60%로 세팅",
  "@계량분석팀장 005930, 000660 기술적 분석 좀",
  "@마켓센티먼트팀장 최근 반도체 업종 뉴스 요약",
  "@사후관리실장 보유 종목 점검",
  "@운용지원실장 최근 사이클 진단하고 내 전략 파라미터 손볼 곳 있나 봐줘",
].forEach((t) => children.push(bullet(t)));

// 12. AI 팀과 분석 사이클
children.push(H1("12. AI 팀과 분석 사이클"));
children.push(P("ArQuant는 10명의 AI 에이전트가 한 사이클마다 다음 역할을 분담합니다. 사이드바의 에이전트 조직도는 FRONT(리서치·운용전략) / MIDDLE(리스크 통제·사후관리) / BACK(실행·인프라 지원) 3구역으로 나뉩니다."));
children.push(table(
  ["에이전트", "역할"],
  [
    ["주식운용실장", "총괄 전략·종목 선정(2패스: 후보 → 최종) · 사장 지시 처리"],
    ["글로벌리서치팀장", "거시·매크로 분석 → 4분할 자산배분 권고 (주식/채권/원자재/현금)"],
    ["계량분석팀장", "후보 종목 정량 해설 (점수는 결정론 엔진이 산출) + 진입/매도가 제시"],
    ["마켓센티먼트팀장", "뉴스 감성·이벤트 분석, 시장(KR/US/공통) 분류"],
    ["채권운용실장", "채권 비중 권고를 채권 ETF 매매로 실현 (국고채·CD금리·미국채 등)"],
    ["원자재운용실장", "원자재 비중 권고를 원유·금·농산물 ETF 매매로 실현"],
    ["포트폴리오기획팀장", "매수 직후 보유 계획(목표가/손절가/기간) 기록 → 매도 판단 때 권고로 상기"],
    ["리스크관리실장", "리스크 게이트(결정론) + DART 공시 기반 2차 재심"],
    ["사후관리실장", "보유 종목 매도/유지/익절·손절 최종 판단 (주식 + 채권·원자재 매도 제안 종합)"],
    ["프롭트레이딩팀장", "주문 실행·체결 결과 보고"],
  ],
  [2400, 6960]));
children.push(P("(+ 운용지원실장 — 사이클 진단과 내 프로필 전략 파라미터 조정 제안. 분석 사이클 밖에서 별도로 동작합니다.)"));
children.push(P("사이클은 대략 다음 순서로 흐릅니다: 글로벌 지수 수집 → 보유종목 조회 → (한국 세션) DART 공시 → 뉴스 분석 → 매크로 분석(4분할 배분) → 주식 후보 선정 + 채권·원자재 슬리브 판단 → 종목 데이터 수집(3년 일봉·수급·분봉) → 퀀트 평가 → 최종 매수 결정 → 보유종목 매도 판단(슬리브 제안 종합) → 주문 초안 조립 → 리스크 검증(결정론 + DART) → 실행(KIS 주문) → 체결 확인·원장 반영 → 체결 보고 → 사이클 리포트."));
children.push(note("AI 비용 절감 구조", "시장 전체에 공통인 분석(매크로 리서치·뉴스 분류)은 관리자 계정이 시간당 1회만 생산하고 다른 계정은 그 결과를 공유받습니다. 내 계정 고유의 판단(종목 선정·매도·리스크)은 각자 수행합니다."));
children.push(H2("12.1 리스크 게이트"));
children.push(P("매수 주문은 다음 결정론 검증을 모두 통과해야 실행됩니다(값은 활성 전략 기준): 단일 종목 비중 한도, 사이클 누적 예산 한도, 예수금 안전마진, 계좌 평가손익이 신규매수 차단선 아래면 전체 매수 중단, 비정상 수량 차단. 이어 매수 종목은 DART 공시·재무 재심(관리종목·거래정지·횡령·대규모 증자/감자·연속적자 등 적신호면 반려)을 받습니다. 매도는 리스크를 줄이는 행위라 통과시킵니다."));
children.push(note("운용지원실장의 역할", "운용지원실장은 코드를 직접 고치거나 서버를 재시작하지 않습니다. 사이클 결과를 진단해 ‘내 계정(프로필)의 전략 파라미터를 어떻게 조정하면 좋을지’ 제안·반영하는 역할만 합니다(전략 탭 토글로 on/off)."));

// 13. API 비용
children.push(H1("13. API 비용 확인"));
children.push(P("상단바의 💵 배지는 로컬 LLM API 사용량을 공식 모델 단가로 계산한 추정 비용입니다."));
children.push(P("표시 단위(시간당 / 일별 / 월별 / 총 누적)는 프로필 모달의 ‘API 비용 표시’에서 고를 수 있습니다(계정별 적용)."));
children.push(new Paragraph({ spacing: { after: 120, line: 312 }, children: [
  new TextRun("💵 배지를 클릭하면 "),
  link("로컬 LLM 서버", "http://127.0.0.1:8080/v1"),
  new TextRun("가 열립니다. 여기서 잔여 크레딧을 확인하고 충전할 수 있습니다."),
]}));
children.push(note("팁", "크레딧이 소진되면 AI 분석이 중단됩니다. 비용 배지를 주기적으로 확인하고 미리 충전해 두세요."));

// 14. 모바일 앱
children.push(H1("14. 모바일 앱 사용"));
children.push(P("모바일 앱은 웹 대시보드와 동일한 서버를 사용하며, 화면도 서버의 웹 화면을 그대로 표시합니다. 따라서 서버에서 기능이나 화면을 수정하면 앱을 다시 설치하지 않아도 다음 실행 시 자동으로 반영됩니다."));
children.push(H2("14.1 로그인"));
children.push(P("앱을 처음 켜면 아이디/비밀번호 로그인 화면이 나옵니다. 로그인하면 그 세션이 내부 화면에 자동 적용됩니다. 웹 화면에서 로그아웃하면 앱도 로그인 화면으로 돌아갑니다."));
children.push(H2("14.2 푸시 알림 (4종 + 백그라운드 지속)"));
children.push(P("앱은 백그라운드에서도 서버와 실시간 연결을 유지하므로, 앱을 닫아 두어도(완전히 종료하지 않는 한) 다음 4가지 알림을 휴대폰으로 받습니다."));
[
  "체결 신청 — 주문이 접수되었을 때",
  "체결 완료 — 매수·매도가 실제로 체결되었을 때",
  "사이클 완료 — 분석 사이클이 끝났을 때",
  "장 마감 — 한국·미국 장이 마감될 때 (당일·누적 수익률 함께 표시)",
].forEach((t) => children.push(bullet(t)));
children.push(P("이 4가지는 ‘프로필 → 모바일 알림 설정’에서 각각 켜고 끌 수 있습니다(계정별). 끈 종류는 휴대폰 알림으로 오지 않으며, 웹 대시보드 통신 로그에는 설정과 무관하게 모두 표시됩니다. 홈 화면 위젯에는 보유 종목·총평가·수익률이 표시되고, 외부 링크(예: 💵 배지)는 기기 기본 브라우저로 열립니다."));
children.push(note("자동 반영 vs 재설치", "화면·기능 수정은 앱을 다시 설치하지 않아도 즉시 반영됩니다. 알림·위젯 같은 네이티브 기능 자체를 바꾼 경우에만 새 APK 설치가 필요합니다."));

// 15. 내 계정 관리
children.push(H1("15. 내 계정 관리"));
children.push(P("상단바의 👤 버튼을 누르면 내 계정 설정 창이 열립니다. 각 항목은 제목을 누르면 펼쳐지는 아코디언입니다. 모든 변경은 보안 감사 기록에 남습니다. 창 하단에 ‘로그아웃’과 ‘회원 탈퇴’ 버튼이 있습니다."));
children.push(H2("15.1 비밀번호 변경"));
children.push(P("현재 비밀번호를 확인한 뒤 새 비밀번호(10자 이상 + 특수문자 1개 이상)로 바꿉니다."));
children.push(H2("15.2 정보 변경 (API 자격증명)"));
children.push(P("로컬 LLM · KIS App Key/Secret · 계좌번호 · 거래 환경을 변경할 수 있습니다. 입력한 값은 실제 호출로 검증됩니다."));
children.push(note("주의", "계정 복구는 ‘계좌번호 + App Secret’으로 본인을 확인합니다. 이 두 값을 바꾸면 이후 복구 시 새 값을 입력해야 합니다."));
children.push(H2("15.3 지시사항 관리 (상시 지시)"));
children.push(P("내 계정에만 적용되는 운용 원칙을 영구 등록할 수 있습니다(예: 방어 자산 우선 등). 주식운용실장이 매 사이클 참고합니다. 추가/삭제가 자유로우며, 삭제한 지시는 서버를 재시작해도 다시 살아나지 않습니다."));
children.push(note("안전", "상시 지시는 AI의 ‘참고 지침’으로만 들어가며, 주문을 막는 결정론적 리스크 한도는 절대 우회하지 못합니다."));
children.push(H2("15.4 API 비용 표시"));
children.push(P("우상단 💵 배지의 합산 단위를 시간당 / 일별 / 월별 / 총 누적 중에서 고릅니다(이 프로필에만 적용)."));
children.push(H2("15.5 모바일 알림 설정"));
children.push(P("휴대폰으로 받을 푸시 4종(체결 신청 / 체결 완료 / 사이클 완료 / 장 마감)을 각각 켜고 끕니다(계정별). 자세한 종류는 14.2를 참고하세요."));
children.push(H2("15.6 피드백 · 버그 제보"));
children.push(P("‘피드백/버그 제보’에서 버그·건의사항을 운영자(사장님)에게 직접 보낼 수 있습니다. 운영자가 확인 후 답글을 달면 같은 화면에서 볼 수 있습니다."));
children.push(num("종류를 고릅니다 — 🐞 버그 제보 / ✨ 기능 요청 / 💬 건의·기타."));
children.push(num("제목과 상세 내용을 적고 ‘제출’을 누릅니다."));
children.push(num("제출 내역은 아래 목록에 쌓이고, 답글이 달리면 그 자리에 함께 표시됩니다(상태: 대기중 → 답변됨)."));
children.push(P("새 답글이 오면 👤 버튼과 이 섹션 제목 옆에 빨간 점/배지가 나타납니다. 화면을 열어 답글을 확인하면 배지는 사라집니다. 내가 보낸 제보는 나만 볼 수 있습니다(운영자는 전체를 봅니다)."));
children.push(H2("15.7 로그아웃 / 회원 탈퇴"));
children.push(P("창 하단 ‘로그아웃’으로 세션을 종료합니다. ‘회원 탈퇴’를 누르고 비밀번호를 다시 입력하면 계정·세션·저장된 API 자격증명·상시 지시가 모두 영구 삭제됩니다(되돌릴 수 없음). 시스템에 ADMIN이 한 명뿐인 상황을 막기 위해 ADMIN 계정은 탈퇴할 수 없습니다."));
children.push(note("보안", "세션 토큰은 7일 유효하며 서버 재시작 후에도 유지됩니다. 세션 쿠키는 HttpOnly+Secure+SameSite로 보호되고, API 비밀키는 서버에 암호화 저장되어 단말로 내려오지 않습니다. 공용 기기에서는 사용 후 반드시 로그아웃하세요."));

// 16. 관리자(ADMIN) 기능
children.push(H1("16. 관리자(ADMIN) 기능"));
children.push(P("ADMIN 계정으로 로그인하면 상단에 🛡️ ADMIN 탭이 추가로 보입니다. 일반 사용자에게는 보이지 않습니다."));
[
  "전역 설정 — 에이전트·뉴스 크롤러 모델(역할명별 입력), 뉴스 크롤 주기(초)를 바꿉니다. 모델 변경은 다음 재시작에 반영, 크롤 주기는 즉시 반영됩니다. 빈칸이면 기본값을 씁니다.",
  "회원 관리 — 전체 회원 목록과 ADMIN 권한 부여/해제. 프로필 창의 ‘회원 관리’에서는 일반 회원 삭제도 가능합니다(본인·다른 ADMIN은 삭제 불가).",
  "피드백/버그 제보 — 전체 사용자의 제보를 모아 보고 답글을 작성합니다. 새 제보가 들어오면 ADMIN 탭에 빨간 점이 표시되고, 답글을 달면 해당 사용자에게 알림이 갑니다.",
].forEach((t) => children.push(bullet(t)));

// 17. FAQ
children.push(H1("17. 자주 묻는 질문 (FAQ)"));
const faq = [
  ["실행을 눌렀는데 거래가 안 일어나요.", "장이 열려 있어야 거래합니다(세션표 5장). 장외에는 뉴스만 수집합니다. 또한 전략의 위험 한도(예: 평가손익이 신규매수 차단선 아래)나 예수금 부족(5,000원 미만)으로 매수가 막혔을 수 있습니다."],
  ["주말·공휴일인데 거래를 안 해요.", "정상입니다. 주말은 자동 휴장이고, 평일에도 개장 5분 후 실시세로 휴장 여부를 확인해 휴장일이면 사이클을 건너뜁니다(5장)."],
  ["장 시간이 아닌데 수익률이 계속 바뀌나요?", "아닙니다. 장 운영시간이 아니면 평가금액 포인트를 기록하지 않고 수익률 탭도 자동 갱신을 멈춰, 마지막 장중 값으로 동결됩니다(7.2)."],
  ["거래 내역을 비우면 승률이 사라지나요?", "아니요. 거래 내역을 비워도 승률·통계는 따로 적립되어 유지되고, 누적수익 곡선도 별도 보존됩니다(8.2)."],
  ["수익률 차트가 평가금액이 아니라 이상해요.", "차트는 ‘평가금액’이 아니라 시작점을 0원으로 둔 누적수익(원)입니다(7.2). 잔고 절대값은 보유종목 탭에서 보세요."],
  ["DART 키는 어디에 넣나요?", "넣지 않습니다. 공시·재무 재심은 시스템이 자체 키로 자동 수행합니다."],
  ["API 비용은 어디서 확인하나요?", "상단 💵 배지와 로컬 LLM 서버의 Billing 메뉴에서 확인합니다."],
  ["AI 분석이 갑자기 멈췄어요.", "로컬 LLM와 잔액, 네트워크 상태를 확인하세요."],
  ["버그를 발견했거나 건의할 게 있어요.", "프로필 창의 ‘피드백/버그 제보’에서 종류를 골라 보내면 운영자가 확인 후 답글을 답니다. 답글이 오면 👤 버튼에 빨간 점이 뜹니다(15.6)."],
  ["아이디나 비밀번호를 잊었어요.", "로그인 화면의 ‘아이디/비밀번호를 잊으셨나요?’를 쓰세요. 가입 때 입력한 계좌번호 + App Secret으로 본인을 확인합니다(3.3). 이메일·문자 인증은 없으니 이 두 값을 꼭 보관하세요."],
  ["@운용지원실장에게 설정을 바꿔달라고 했는데 안 바뀌어요.", "운용지원실장은 코드·설정을 직접 바꾸지 않고 ‘진단·제안’만 합니다. 전략 값은 10장 ‘전략 커스터마이즈’에서 직접 적용하세요."],
  ["계정·로그인이 서버 재시작 후에도 유지되나요?", "네. 계정·세션(7일)은 서버 디스크에 영속되어 재시작·코드 갱신 후에도 유지됩니다."],
  ["증권사 앱 잔고와 ArQuant 평가액이 조금 달라요.", "정상일 수 있습니다. ArQuant는 자체 실거래 원장으로 평가하는데, 증권사 집계값은 해외 매도대금 결제 지연(T+2)·전산 글리치로 일시적으로 다르게 보일 수 있습니다(7.1). 결제가 끝나면 대체로 일치합니다. 괴리 경고가 반복되면 입출금/수동매매 여부를 확인하세요."],
  ["입출금(이체)이나 수동 매매를 했더니 수익률이 이상해요.", "원장은 ArQuant의 체결만 알기 때문에 외부 입출금·수동 매매는 반영되지 않습니다. 그런 일이 있으면 원장 재동기화(/api/ledger/reseed)가 필요합니다 — 괴리가 생기면 시스템이 30분 내 경고 알림을 보냅니다."],
  ["시간외(프리/애프터)에 어떤 종목은 거래를 안 해요.", "NXT 대체거래소에 상장되지 않은 종목입니다. 한 번 거부되면 시스템이 자동으로 기억해(전 계정 공유) 시간외에는 시도하지 않고 정규장에서만 거래합니다(5장)."],
];
faq.forEach(([q, a]) => {
  children.push(new Paragraph({ spacing: { before: 100, after: 40, line: 300 },
    children: [new TextRun({ text: "Q. " + q, bold: true })] }));
  children.push(new Paragraph({ spacing: { after: 120, line: 300 },
    children: [new TextRun("A. " + a)] }));
});

// 18. 문제 해결
children.push(H1("18. 문제 해결"));
children.push(table(
  ["증상", "조치"],
  [
    ["로그인이 안 됨", "아이디/비밀번호 확인(비번은 10자 이상 + 특수문자 1개). 잊었다면 ‘아이디/비밀번호 찾기’(계좌번호+App Secret)로 복구(3.3)."],
    ["잔고가 0으로 보임", "🔄 새로고침 클릭. KIS 키/계좌번호·거래 환경(실전/모의)이 정확한지 확인."],
    ["주문이 계속 실패", "거래 환경(실전/모의)과 키 종류가 일치하는지 확인. 모의 키에 실전 환경을 고르면 주문이 실패합니다(2.2)."],
    ["거래 내역이 비어 있음", "아직 체결이 없거나 🗑️ 비우기로 초기화된 상태입니다(승률·곡선은 유지)."],
    ["모바일에서 화면이 안 뜸", "네트워크 확인 후 앱 재시작. 로그인 화면이 나오면 다시 로그인."],
  ],
  [2600, 6760]));

// 19. 안전 유의사항
children.push(H1("19. 안전 유의사항 / 면책"));
[
  "ArQuant는 실제 자금으로 실거래를 수행합니다(기본 LIVE 모드). 손실 가능성이 있으며 모든 투자 책임은 사용자에게 있습니다.",
  "처음에는 모의투자 또는 보수적인 설정값(작은 주문 비율·빠른 손절)과 작은 예산 비율로 시작해 동작을 충분히 관찰하세요.",
  "AI 분석은 외부 모델·데이터에 의존하므로 오류·지연·중단이 발생할 수 있습니다.",
  "API 키와 계좌 정보(특히 App Secret)는 타인과 공유하지 마세요 — 계정 복구 열쇠이기도 합니다.",
].forEach((t) => children.push(bullet(t)));
children.push(P("본 문서는 현재 시스템(웹 대시보드·트레이딩 엔진) 동작을 기준으로 작성된 사용자용 안내서입니다."));

// ── build ─────────────────────────────────────────────────
const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: FONT, color: "1F3B57" },
        paragraph: { spacing: { before: 320, after: 200 }, outlineLevel: 0,
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "1F3B57", space: 4 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: FONT, color: "2E5276" },
        paragraph: { spacing: { before: 220, after: 140 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: FONT, color: "44607A" },
        paragraph: { spacing: { before: 160, after: 100 }, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [
      { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 620, hanging: 320 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 1040, hanging: 320 } } } } ] },
    { reference: "steps", levels: [
      { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 620, hanging: 320 } } } } ] },
  ] },
  sections: [{
    properties: { page: {
      size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({
      spacing: { after: 0 }, border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "C9D3DF", space: 6 } },
      children: [new TextRun({ text: "ArQuant v1.0 사용설명서", size: 16, color: "8A99AB" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 60 },
      children: [new TextRun({ text: "— ", size: 16, color: "8A99AB" }),
        new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "8A99AB" }),
        new TextRun({ text: " —", size: 16, color: "8A99AB" })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("사용설명서.docx", buf);
  console.log("WROTE 사용설명서.docx  (" + buf.length + " bytes)");
});
