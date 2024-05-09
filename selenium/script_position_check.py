from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import base64
import time
import openai
import pandas as pd
import os.path
# import gspread
# from gspread_dataframe import set_with_dataframe


# json_file_path = 'C:/Users/weven/Desktop/AITEST/vision_ai/selenium/mailing-421207-c0e8e568b22a.json'
# gc = gspread.service_account(json_file_path)
# spreadsheet_url = 'https://docs.google.com/spreadsheets/d/135wSfAC7Rp9ct3bbvxC090AVqy4La4c5sTDl1B0l26A/edit#gid=0'
# doc = gc.open_by_url(spreadsheet_url)
# worksheet_data = doc.worksheet('태그')

### ==============================
# Please Don't forget to input OpenAI API key
### ==============================
# openai.api_key =


def make_dataframe_to_excel(col_1: list, col_2: list, url: str):
    print("start excel download and google sheet")
    
    # 저장할 엑셀 파일이름 설정
    excel_file_name = 'script_shoping_result.xlsx'
    
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

        print("set_with_dataframe")

    else:
        # 처음 엑셀 파일을 생성할 때
        with pd.ExcelWriter(excel_file_name, engine='openpyxl') as writer:
            df = pd.DataFrame({"Script": col_1, "Result_from_GPT": col_2})
            df["Script"] = df["Script"].str.replace('\n', '')
            # 새로운 시트에 저장
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        print("set_with_dataframe")

    # if os.path.isfile("./" + csv_file_name):
    #     previos_df = pd.read_csv("./" + csv_file_name, index_col=False)
    #     new_df = pd.DataFrame({"Script" : ["new_script"], "Result_from_GPT" : ["new_gpt_result"]})
    #     current_df = pd.DataFrame({"Script" : col_1, "Result_from_GPT" : col_2})
    #     current_df = current_df.replace('\\n', '')
        
    #     modify_current_df = pd.concat([new_df, current_df])
        
    #     combined_df = pd.concat([previos_df, modify_current_df])
    #     combined_df.to_csv(csv_file_name, encoding = 'utf-8-sig', index=False, sheet_name=url)
        
    # else:
    #     df = pd.DataFrame({"Script" : col_1, "Result_from_GPT" : col_2})
    #     modify_df = df.replace('\\n', '')
        
    #     modify_df.to_csv(csv_file_name, encoding = 'utf-8-sig', index=False, sheet_name=url)

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
    word = openai_script_analysis(script_list)
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

def openai_script_analysis(data):
    ##### NOTE : 만약 긴 스크립트들이 의미없다고 생각될 경우 아래 max_tokens 조건 살리기. 
    #            아마 스크립트가 길어서 https://loveparis.net/ 에서 시간이 오래 소요됨
    # token_per_char=0.5
    # estimated_tokens = len(data) * token_per_char
    # max_tokens = 1000 - int(estimated_tokens)
    # if max_tokens < 1:
    #     max_tokens = 1
    if data != '' or data is None:
        word = []
        try:
            for i in data:
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    
                    # innerHTML이면 그걸 그대로 해석 / javascript면 어떤 기능의 라이브러리를 호출하는지 알려주기
                    messages=[
                        {"role": "system", "content": """Your role is a script interpreter. Please create something that interprets a script 
                         and extracts just the values in a word tag format to understand what role the interpreted script serves. 
                         And if the input is only javascript file, Tell me the functionality of the library that the script calls. 
                         결과는 한글로 해석해줘"""},
                        {"role": "user", "content": i }
                    ],
                    temperature=0.5
                )
                # Access the response data using a method or property
                improved_text = response.choices[0].message.content
                word.append(improved_text)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        print(word)
        return word
    else:
        return "Too Long"    


pension_site_list = ['https://monacampark.co.kr/', 'https://myeongranghouse.modoo.at/', 'http://www.sbpension.net/', 'https://durbanhill.modoo.at/', 'https://www.welchon.com/', 'http://www.marinuspoolvila.com/', 'http://www.hileisure.net/', 'http://xn--jk1bl1ki6bl6gvuanj892i.com/', 'https://sk0809h.modoo.at/', 'http://boracaypension.com/']
hosital_site_list = ['http://www.brjkmc.co.kr/b/', 'https://www.daprs.com/', 'https://www.idhospital.com/?']
shoping_site_list = ["https://attrangs.co.kr/", "https://loveparis.net/", "https://sonyunara.com/", "https://www.canmart.co.kr/", "https://classic-blanc.com/"]
result_list1 = []
result_list2 = []
result_list3 = []

# , 'http://chukchukdisk.com/', 'https://www.avenueclinic.co.kr/', 'https://drbong.modoo.at/', 'https://youngclinic.imweb.me/', 'https://www.drnoh.kr/', 'http://www.welldental.co.kr/', 'https://kduro.com/']

if __name__ == '__main__':
    # for url in pension_site_list:
    #     result = script_position_check(url)
    #     result_list2.extend(result)  # 사용 시 script_position_check의 return값 조정 후 extend() -> concat()
    
    # for url in hosital_site_list:
    #     result = script_position_check(url)
    #     result_list2.extend(result)  # 사용 시 script_position_check의 return값 조정 후 extend() -> concat()
 
    for url in shoping_site_list:
        result = script_position_check(url)
    #    result_list3.extend(result)  # 사용 시 script_position_check의 return값 조정 후 extend() -> concat()
    
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
    