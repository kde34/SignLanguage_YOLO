import requests
import datetime
import uuid
import hmac
import hashlib

def send_solapi_sms(to_number, translated_word):
    """솔라피를 이용해 번역된 수어를 문자로 전송하는 함수"""
    
    # ⚠️ 여기에 솔라피 홈페이지에서 발급받은 정보를 쏙쏙 넣어주세요!
    API_KEY = "--"
    API_SECRET = "--"
    FROM_NUMBER = "--" # 예: 01012345678 (하이픈 빼고 숫자만!)
    
    # 솔라피 공식 암호화(HMAC) 인증 과정
    date = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    salt = str(uuid.uuid1().hex)
    data = date + salt
    signature = hmac.new(API_SECRET.encode(), data.encode(), hashlib.sha256).hexdigest()
    auth_header = f'HMAC-SHA256 apiKey={API_KEY}, date={date}, salt={salt}, signature={signature}'
    
    # 💌 스마트폰으로 날아갈 문자 내용 구성!
    text_message = f"[9조 실시간 수어 번역]\n방금 인식된 수어는 '{translated_word}' 입니다!"
    
    try:
        response = requests.post(
            'https://api.solapi.com/messages/v4/send',
            headers={'Authorization': auth_header, 'Content-Type': 'application/json'},
            json={
                'message': {
                    'to': to_number,
                    'from': FROM_NUMBER,
                    'text': text_message
                }
            }
        )
        
        if response.status_code == 200:
            print(f"📱 문자 전송 성공! ➔ 내용: {translated_word}")
        else:
            print(f"❌ 문자 전송 실패 ➔ 에러코드: {response.text}")
            
    except Exception as e:
        print(f"⚠️ 문자 전송 중 오류 발생: {e}")
