"""ArcTrade 크립토 — CryptoBars 아카이브 위에서 도는 통계적 차익거래(Avellaneda-Lee).

KRX 경로(core/·web/autofolio)와 완전히 분리돼 있다. 공유하는 건 OU/s-score 수학
(core.analytics.ou_score) 하나뿐이다. 24/7 시장이라 세션·동시호가·일 경계가 전부 없고,
perp 이라 숏이 가능해 논문 그대로의 롱숏 시장중립을 쓴다(KRX 는 현물 롱온리였다).
"""
