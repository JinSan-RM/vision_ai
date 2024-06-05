from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import base64
import time
import pandas as pd
import os


def make_dataframe_to_excel(col_1: list, col_2: list, url: str):
    print("start excel download and google sheet")
    
    # 저장할 엑셀 파일이름 설정
    excel_file_name = 'script_pension_result.xlsx'
    
    # URL을 시트 이름으로 변환하기 위한 정리 작업
    sheet_name = url.replace('http://', '').replace('https://', '').replace('/', '_').replace('.', '_')


    if os.path.isfile(excel_file_name):
        
        # 기존 엑셀 파일이 있을 때 처리
        with pd.ExcelWriter(excel_file_name, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            # 새로운 데이터를 DataFrame에 저장
            current_df = pd.DataFrame({"Script": col_1, "Result_from_GPT": col_2})
            current_df["Script"] = current_df["Script"].str.replace('\n', '')
            # 새 시트에 작성
            current_df.to_excel(writer, sheet_name=sheet_name, index=False)
        print("set_with_dataframe")

    else:
        # 처음 엑셀 파일을 생성할 때
        with pd.ExcelWriter(excel_file_name, engine='openpyxl') as writer:
            df = pd.DataFrame({"Script": col_1, "Result_from_GPT": col_2})
            df["Script"] = df["Script"].str.replace('\n', '')
            # 새로운 시트에 저장
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        print("set_with_dataframe")

def script_position_check(url : str):

    print(f"Accessing {url}")
    driver = create_chrome_driver()
    driver.get(url)
    
    driver.set_window_size(1920, 1080)
    before_h = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.set_window_size(1920, 1080)
        time.sleep(5)

        driver.execute_script("window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth'});")

        time.sleep(5)

        after_h = driver.execute_script("return document.body.scrollHeight")

        driver.execute_script(f"window.scrollTo(0, {after_h} - 1000);")
        driver.execute_script(f"window.scrollTo(0, {after_h});")
        time.sleep(5)
        driver.execute_script("window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth'});")
        driver.execute_script("window.scrollTo({ top: document.body.scrollHeight + 500, behavior: 'smooth'});")
        time.sleep(5)
        
        after_h = driver.execute_script("return document.body.scrollHeight")
        
        if after_h == before_h:
            break
        before_h = after_h
            
    print("scroll finished")
    
    script_list = []
    script_elements = driver.find_elements(By.TAG_NAME, 'script')
    
    # script의 innerHTML 및 src 감지하여 append
    for script in script_elements:
        innerHTML = script_list.append(script.get_attribute('innerHTML'))
        if innerHTML == None:
            script_list.append(script.get_attribute('src'))
            
    print(script_list, "<<<====script_list")
    script_list = list(filter(None, script_list))
    driver.quit()
    word = script_list
    print(word, "<=====word")
    make_dataframe_to_excel(script_list, word, url)
    
    return "sucess"


def create_chrome_driver():
    options = webdriver.ChromeOptions()
    user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36"
    options.add_argument('user-agent=' + user_agent)
    options.add_argument("--lang=ko-KR")
    # options.add_argument('--headless')
    # options.add_argument('--no-sandbox')
    options.add_argument("--start-maximized")
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_experimental_option('excludeSwitches', ['disable-popup-blocking'])
    
    # 드라이버 인스턴스 생성
    driver = webdriver.Chrome(options=options)
    return driver    



pension_site_list = ['https://monacampark.co.kr/', 'https://myeongranghouse.modoo.at/', 'http://www.sbpension.net/', 'https://durbanhill.modoo.at/', 'https://www.welchon.com/', 'http://www.marinuspoolvila.com/', 'http://www.hileisure.net/', 'http://xn--jk1bl1ki6bl6gvuanj892i.com/', 'https://sk0809h.modoo.at/', 'http://boracaypension.com/']
hosital_site_list = ['http://www.brjkmc.co.kr/b/', 'https://www.daprs.com/']
# hosital_site_list = ['http://www.brjkmc.co.kr/b/', 'https://www.daprs.com/', 'https://www.idhospital.com/?', 'http://chukchukdisk.com/', 'https://www.avenueclinic.co.kr/', 'https://drbong.modoo.at/', 'https://youngclinic.imweb.me/', 'https://www.drnoh.kr/', 'http://www.welldental.co.kr/', 'https://kduro.com/']
shoping_site_list = ["https://attrangs.co.kr/", "https://loveparis.net/", "https://sonyunara.com/", "https://www.canmart.co.kr/", "https://classic-blanc.com/"]
result_list1 = []
result_list2 = []
result_list3 = []

if __name__ == '__main__':
    # for url in pension_site_list:
    #     result = script_position_check(url)
    #     result_list2.extend(result)  # 각 사이트 결과를 result_list1에 추가
    
    for url in hosital_site_list:
        result = script_position_check(url)
        # result_list2.extend(result)  # 각 사이트 결과를 result_list2에 추가
 
    # for url in shoping_site_list:
    #     result = script_position_check(url)
    #     result_list3.extend(result)  # 각 사이트 결과를 result_list3에 추가
    
    
    # ======== NOTE : 아래 부분을 활성화하려면 script_position_check의 return 값을 
    #                 Dataframe으로 받고 extend() -> concat()로 변환하여 사용
    #                 (※ 한 파일에 넣을거면 리소스 측면에서 이게 좋음. 현재는 사이트별로 각각 탭에 저장중)
    # 모든 결과를 하나의 엑셀 파일로 저장
    # df1 = pd.DataFrame(result_list1, columns=['Script Content'])
    # df2 = pd.DataFrame(result_list2, columns=['Script Content'])
    # df3 = pd.DataFrame(result_list3, columns=['Script Content'])
    
    # df1.to_excel(f'{result_list1}output.xlsx', index=False)
    # df2.to_excel(f'{result_list2}output.xlsx', index=False)
    # df3.to_excel(f'{result_list3}output.xlsx', index=False)
    
    print("Results saved to Excel.")