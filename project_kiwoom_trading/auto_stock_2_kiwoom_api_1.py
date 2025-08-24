# -*- coding: utf-8 -*-
from pykiwoom.kiwoom import Kiwoom

# ===== 사용자 입력 =====
ACCOUNT_INDEX = 0   # N번째 계좌 사용
ACCOUNT_PW = ""     # 계좌 비밀번호(빈 값이면 아래가 비밀번호 입력창을 띄움)
QUERY_TYPE = 2      # 조회구분: 1=추정조회, 2=일반조회
# ======================

def get_deposit(kw: Kiwoom, account_no: str, password: str, query_type: int) -> int:
    df = kw.block_request(
        "opw00001",               # 예수금상세현황요청
        계좌번호=account_no,
        비밀번호=password,         # 비밀번호 필요
        비밀번호입력매체구분="00",  # 00=키보드(일반)
        조회구분=query_type,        # 1 추정 / 2 일반
        output="예수금상세현황",
        next=0
    )
    if df.empty:
        raise RuntimeError("예수금 조회 실패")
    val = str(df.loc[0, "예수금"]).replace(",", "").strip()
    return int(val or 0)

if __name__ == "__main__":
    kw = Kiwoom()
    kw.CommConnect(block=True)

    # 비번을 코드에 안 넣었으면, 비밀번호 입력창을 먼저 띄워 사용자 입력 받기
    if not ACCOUNT_PW:
        # 키움 제공 함수: 계좌비밀번호 입력창 호출
        kw.ocx.dynamicCall("KOA_Functions(QString, QString)", "ShowAccountWindow", "")
        # 여기서 사용자가 창에 비번 입력 후 확인을 눌러야 다음 요청이 성공합니다.

    accounts = kw.GetLoginInfo("ACCNO")
    if not accounts:
        print("계좌 없음")
    else:
        acc = accounts[ACCOUNT_INDEX]
        deposit = get_deposit(kw, acc, ACCOUNT_PW, QUERY_TYPE)
        print(f"[{acc}] 예수금: {deposit:,} 원")
