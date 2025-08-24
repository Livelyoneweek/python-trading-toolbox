# -*- coding: utf-8 -*-
"""
Qt Designer로 만든 gui.ui 로드:
- 최상위가 QDialog든 QMainWindow든 모두 동작 (uic.loadUiType 사용)
- code_list: QLineEdit (콤마로 코드 입력: "005930, 005380")
- button_start: QPushButton
- button_stop: QPushButton
- textboard: QTextBrowser
"""

import sys
from datetime import datetime
from typing import List, Dict

from PyQt5 import uic
from PyQt5.QtCore import QTimer
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
from pykiwoom.kiwoom import Kiwoom

# ===== 설정 =====
UI_PATH = "gui.ui"
UPDATE_INTERVAL_MS = 1000   # 1초마다 갱신
REQUEST_GAP_MS   = 250      # TR 사이 간격(과호출 방지)
CLEAR_LOG_ON_STOP = False   # 중단 시 로그를 지울지 여부
# =================

def _clean_num(v) -> int:
    s = str(v).replace("+", "").replace("-", "").replace(",", "").strip()
    return int(s or 0)

# ui의 FormClass, BaseClass를 동적으로 가져옴 (QDialog/QMainWindow 모두 호환)
FormClass, BaseClass = uic.loadUiType(UI_PATH)

class MainWindow(BaseClass, FormClass):
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # <- gui.ui에 정의된 위젯들이 self로 붙음

        # 시그널
        self.button_start.clicked.connect(self.on_start)
        self.button_stop.clicked.connect(self.on_stop)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_tick)

        self.kw: Kiwoom | None = None
        self.codes: List[str] = []
        self.code_name_cache: Dict[str, str] = {}

        self.set_running(False)

    # 상태 전환
    def set_running(self, running: bool):
        self.button_start.setEnabled(not running)
        self.button_stop.setEnabled(running)
        self.code_list.setEnabled(not running)

    # 시작
    def on_start(self):
        raw = self.code_list.text().strip()
        if not raw:
            self._append_line("[알림] 종목코드를 콤마(,)로 구분하여 입력하세요. 예) 005930, 005380")
            return
        # 입력 파싱(공백 제거, 중복 제거)
        self.codes = list(dict.fromkeys([c.strip() for c in raw.split(",") if c.strip()]))
        if not self.codes:
            self._append_line("[알림] 유효한 종목코드가 없습니다.")
            return

        try:
            if self.kw is None:
                self.kw = Kiwoom()
                self.kw.CommConnect(block=True)
        except Exception as e:
            self._append_line(f"[오류] 로그인 실패: {e}")
            self.kw = None
            return

        # 종목명 캐시
        self.code_name_cache.clear()
        for code in self.codes:
            try:
                df = self.kw.block_request("opt10001", 종목코드=code, output="주식기본정보", next=0)
                self.code_name_cache[code] = ("" if df.empty else str(df.loc[0, "종목명"]).strip())
            except Exception:
                self.code_name_cache[code] = ""
            QTest.qWait(REQUEST_GAP_MS)

        if CLEAR_LOG_ON_STOP:
            self.textboard.clear()

        self.set_running(True)
        self.timer.start(UPDATE_INTERVAL_MS)

    # 중단
    def on_stop(self):
        self.timer.stop()
        self.set_running(False)
        self.codes.clear()
        self.code_name_cache.clear()
        if CLEAR_LOG_ON_STOP:
            self.textboard.clear()

    # 주기 처리
    def on_tick(self):
        if not self.kw or not self.codes:
            return
        now = datetime.now().strftime("%H:%M:%S")
        lines = []
        for code in self.codes:
            try:
                df = self.kw.block_request("opt10001", 종목코드=code, output="주식기본정보", next=0)
                if df.empty:
                    lines.append(f"[{now}] [{code}] [-] [-]")
                else:
                    name = self.code_name_cache.get(code) or str(df.loc[0, "종목명"]).strip()
                    price = _clean_num(df.loc[0, "현재가"])
                    lines.append(f"[{now}] [{code}] [{name}] [{price:,}]")
            except Exception as e:
                lines.append(f"[{now}] [{code}] [에러] [{e}]")
            QTest.qWait(REQUEST_GAP_MS)
        if lines:
            self._append_line("\n".join(lines))

    def _append_line(self, s: str):
        self.textboard.append(s)
        bar = self.textboard.verticalScrollBar()
        bar.setValue(bar.maximum())

    # 종료 정리 (세그폴트 방지)
    def closeEvent(self, event):
        try:
            self.timer.stop()
            if self.kw is not None:
                for sig in ("OnReceiveTrData", "OnReceiveRealData", "OnEventConnect"):
                    try:
                        getattr(self.kw.ocx, sig).disconnect()
                    except Exception:
                        pass
                self.kw.ocx.deleteLater()
        except Exception:
            pass
        super().closeEvent(event)

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    w = MainWindow()
    w.setWindowTitle("현재가 모니터")
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

