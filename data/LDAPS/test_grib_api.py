import requests
from datetime import datetime, timedelta

# GRIB 파일 다운로드 API 테스트
current_auth_key = "5LGEH7S2SVSxhB-0tnlUlA"

def test_grib_download():
    # 여러 날짜 시도 (더 오래된 날짜들)
    test_dates = [
        "2024090300",  # 2024년 9월 3일 00시
        "2024080300",  # 2024년 8월 3일 00시 
        "2023100300",  # 2023년 10월 3일 00시
    ]
    
    # LDAPS 변수들 (올바른 변수명)
    variables = ["tmpr", "lspr"]  # 기온, 강수량 (노트북에서 확인된 변수명)
    
    for date in test_dates:
        for band in variables:
            url = f'https://apihub.kma.go.kr/api/typ06/url/nwp_vars_down.php?nwp=l015&sub=unis&vars={band}&tmfc={date}&ef=1&dataType=GRIB&authKey={current_auth_key}'
        
            print(f"\n=== Testing {band} variable on {date} ===")
            print(f"URL: {url}")
        
            try:
                r = requests.get(url, timeout=(10, 30))
                print(f"Status: {r.status_code}")
                print(f"Headers: {dict(r.headers)}")
                
                if r.status_code == 200:
                    content_type = r.headers.get('content-type', '')
                    if 'application/json' in content_type:
                        try:
                            js = r.json()
                            print(f"JSON Response: {js}")
                        except:
                            print(f"Response text: {r.text[:500]}")
                    elif 'application/octet-stream' in content_type or 'grib' in content_type.lower():
                        print(f"GRIB file received! Size: {len(r.content)} bytes")
                        # 파일로 저장할 수 있음
                        # with open(f"test_{band}_{date}.grb", "wb") as f:
                        #     f.write(r.content)
                        return True  # 성공시 종료
                    else:
                        print(f"Response text: {r.text[:500]}")
                        if "file not exist" not in r.text:
                            return True  # 다른 성공 응답
                else:
                    print(f"Error: {r.status_code}")
                    print(f"Response: {r.text[:500]}")
                    
            except Exception as e:
                print(f"Request error: {e}")
                
    return False  # 모든 시도 실패

if __name__ == "__main__":
    test_grib_download()