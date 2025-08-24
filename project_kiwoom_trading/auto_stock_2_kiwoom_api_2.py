# -*- coding: utf-8 -*-
"""
키움 OpenAPI+ (pykiwoom)로 opt10001(주식기본정보요청) 조회 스크립트
- 상단 STOCKS, FIELDS만 수정해서 사용
- FIELDS에 적힌 컬럼만 출력
- 종료 시 QApplication/시그널 정리로 세그폴트 방지
"""

import sys
from typing import Dict, List, Any
from PyQt5.QtWidgets import QApplication
from pykiwoom.kiwoom import Kiwoom
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# [1] 조회 대상 종목/코드 (원하는 항목으로 자유롭게 수정)
STOCKS: Dict[str, str] = {
    "삼성전자": "005930",
    "현대차":   "005380",
}

# [2] 보고 싶은 정보 컬럼 리스트 (원하는 항목으로 자유롭게 수정)
#    ─ opt10001(주식기본정보요청)에서 사용 가능한 대표 컬럼 목록 (주석)
#    ['종목코드','종목명','결산월','액면가','자본금','상장주식','신용비율','연중최고','연중최저',
#     '시가총액','시가총액비중','외인소진률','대용가','PER','EPS','ROE','PBR','EV','BPS',
#     '매출액','영업이익','당기순이익','250최고','250최저','시가','고가','저가','상한가','하한가',
#     '기준가','예상체결가','예상체결수량','250최고가일','250최고가대비율','250최저가일','250최저가대비율',
#     '현재가','대비기호','전일대비','등락율','거래량','거래대비','액면가단위','유통주식','유통비율']
FIELDS: List[str] = ["현재가", "시가", "고가", "저가", "전일대비", "등락율", "거래량", "시가총액"]

# [옵션] 숫자 파싱 여부 (콤마/부호 제거하여 int/float로 변환)
PARSE_NUMBERS: bool = True
# ─────────────────────────────────────────────────────────────────────────────


def _clean_number(val: Any):
    """문자열 숫자(+, -, %, ,)를 제거해 int/float로 변환. 실패 시 원본 반환."""
    s = str(val).strip()
    if s == "" or s == "-":
        return 0
    # 등락율 같은 퍼센트 기호 제거
    s = s.replace("%", "")
    # 기호/콤마 제거(대비 부호는 기호 컬럼에서 별도 제공되니 제거해도 무방)
    s = s.replace(",", "").replace("+", "").strip()
    # 정수/실수 판별
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return val


def get_basic_info(kw: Kiwoom, code: str, fields: List[str]) -> pd.Series:
    """opt10001 호출 후, 원하는 컬럼만 반환 (Series)"""
    df = kw.block_request(
        "opt10001",
        종목코드=code,
        output="주식기본정보",
        next=0
    )
    if df.empty:
        raise RuntimeError(f"응답이 비어 있습니다. (code={code})")

    row = df.loc[0]
    # 존재하는 컬럼만 필터
    cols = [c for c in fields if c in row.index]
    result = row[cols].copy()

    if PARSE_NUMBERS:
        # 문자열 수치 -> 숫자 변환 (일부 텍스트 컬럼은 그대로 유지)
        keep_as_is = {"종목명", "대비기호", "250최고가일", "250최저가일", "액면가단위"}
        for c in result.index:
            if c not in keep_as_is:
                result[c] = _clean_number(result[c])

    return result


def main() -> int:
    # QApplication 명시 생성(이미 있으면 재사용)
    app = QApplication.instance() or QApplication(sys.argv)

    kw = Kiwoom()
    kw.CommConnect(block=True)  # 로그인 완료까지 대기

    try:
        for name, code in STOCKS.items():
            info = get_basic_info(kw, code, FIELDS)
            # 보기 좋게 출력
            print(f"\n[{name}({code})]")
            # 정렬된 표 형태
            max_key = max(len(k) for k in info.index) if len(info.index) else 0
            for k, v in info.items():
                print(f"{k:<{max_key}} : {v}")
        return 0
    finally:
        # 깔끔한 종료(세그폴트 예방)
        try:
            for sig in ("OnReceiveTrData", "OnReceiveRealData", "OnEventConnect"):
                try:
                    getattr(kw.ocx, sig).disconnect()
                except Exception:
                    pass
            kw.ocx.deleteLater()
        except Exception:
            pass
        app.processEvents()
        app.quit()


if __name__ == "__main__":
    sys.exit(main())
