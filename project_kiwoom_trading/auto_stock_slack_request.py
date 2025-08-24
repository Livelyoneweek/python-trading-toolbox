import requests

def send_slack_message(webhook_url, message):
    payload = {'text': message}
    response = requests.post(webhook_url, json=payload)
    return response.text
# 슬랙웹훅URL
webhook_url = "~~~"
# 보낼메시지
message = '안녕하세요, 슬랙 채널에 메시지를 보냅니다!'
# 메시지보내기
send_slack_message(webhook_url, message)