
# 🏝️ 남씨표류기 Weekly Auto Scanner

주봉 기준으로 미국주식 + 크립토를 자동 스캔하고,
**가장 최근 신호가 💎 전환매수인 종목만** 남깁니다.

또 지난주 결과와 비교해:

- `🆕 신규` = 이번 주 새로 조건에 들어온 종목
- `✅ 유지` = 지난주에도 조건에 있었고 아직 다른 신호로 바뀌지 않은 종목

으로 나눕니다.

## 비용

- OpenAI API: 사용 안 함
- 유료 데이터: 사용 안 함
- GitHub Actions: 개인 공개 저장소 기준 무료 범위에서 사용 가능
- Telegram: 무료
- 토큰 사용량: 0

## 갤럭시에서 쓰는 방법

### 1) GitHub 새 저장소 만들기
GitHub 앱/브라우저에서 새 repository를 하나 만듭니다.

### 2) 이 압축파일 내용 전체 업로드
아래 파일/폴더가 저장소 최상단에 있어야 합니다.

- scanner.py
- requirements.txt
- .github/workflows/weekly.yml
- reports/
- state/

### 3) GitHub Pages 켜기
Repository → Settings → Pages → Source를 **GitHub Actions** 로 선택합니다.

### 4) 첫 실행
Repository → Actions → `남씨표류기 Weekly Scanner`
→ `Run workflow`

성공하면 매주 월요일 한국시간 오전 9:20에 자동 실행됩니다.

Pages 주소는 보통:
`https://내아이디.github.io/저장소이름/`

이 주소를 갤럭시 홈 화면에 추가하면 앱처럼 볼 수 있습니다.

## Telegram 알림까지 받고 싶다면

Telegram에서 BotFather로 봇을 만들고 bot token을 받은 뒤,
GitHub Repository → Settings → Secrets and variables → Actions 에

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

두 Secret을 추가하면 됩니다.

Secret을 안 넣어도 스캐너/웹페이지는 정상 작동합니다.

## 현재 데이터 소스

- 미국주식 심볼: Nasdaq Trader Symbol Directory
- 미국 주봉 가격: Yahoo Finance / yfinance
- 크립토: Bybit Spot USDT 거래대금 상위 300개 / CCXT

## PineScript 반영 로직

- RSI 14
- RSI Signal EMA 12
- Oversold < 30
- Overbought > 70
- Reclaim 60
- RSI가 30 미만에 들어간 이력이 있는 상태에서
  RSI가 Signal EMA를 골든크로스 → 💎 전환매수
- 💎 / 🧊 / 🔥 / 😮 중 가장 최근 신호가 💎일 때만 통과
- 진행 중인 이번 주 주봉은 제외

## 결과 파일

- `reports/latest.html` : 갤럭시 웹페이지
- `reports/us_latest.csv`
- `reports/crypto_latest.csv`

## 주의

Yahoo Finance / Bybit와 TradingView는 데이터 공급처 및 주봉 경계 차이 때문에
일부 종목에서 신호가 1봉 정도 다를 수 있습니다.
실제 투자 전 TradingView 차트에서 결과를 대조하세요.
