from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import base64
import time
import openai
import pandas as pd
import os.path


def make_dataframe_to_excel(col_1: list, col_2: list, url: str):
    excel_file_name = 'script_pension_result.xlsx'
    # URL을 시트 이름으로 변환하기 위한 정리 작업
    sheet_name = url.replace('http://', '').replace('https://', '').replace('/', '_').replace('.', '_')

    if os.path.isfile(excel_file_name):
        # 기존 엑셀 파일이 있을 때 처리
        with pd.ExcelWriter(excel_file_name, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            # 새로운 데이터를 DataFrame에 저장
            current_df = pd.DataFrame({"Script": col_1, "Result_from_GPT": col_2})
            current_df = current_df.replace('\n', '')
            # 새 시트에 작성
            current_df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        # 처음 엑셀 파일을 생성할 때
        with pd.ExcelWriter(excel_file_name, engine='openpyxl') as writer:
            df = pd.DataFrame({"Script": col_1, "Result_from_GPT": col_2})
            modify_df = df.replace('\n', '')
            # 새로운 시트에 저장
            modify_df.to_excel(writer, sheet_name=sheet_name, index=False)
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
    for script in script_elements:
        script_list.append(script.get_attribute('innerHTML'))
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
    # token_per_char=0.5
    # estimated_tokens = len(data) * token_per_char
    # max_tokens = 100 - int(estimated_tokens)
    # if max_tokens < 1:
    #     max_tokens = 1
    if data != '' or data is None:
        word = []
        try:
            for i in data:
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    messages=[
                        {"role": "system", "content": "Your role is a script interpreter. Please create something that interprets a script and extracts just the values in a word tag format to understand what role the interpreted script serves. 결과는 한글로 해석해줘"},
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
        return None    


pension_site_list = ['https://monacampark.co.kr/', 'https://myeongranghouse.modoo.at/', 'http://www.sbpension.net/', 'https://durbanhill.modoo.at/', 'https://www.welchon.com/', 'http://www.marinuspoolvila.com/', 'http://www.hileisure.net/', 'http://xn--jk1bl1ki6bl6gvuanj892i.com/', 'https://sk0809h.modoo.at/', 'http://boracaypension.com/']
hosital_site_list = ['http://www.brjkmc.co.kr/b/', 'https://www.daprs.com/', 'https://www.idhospital.com/?', 'http://chukchukdisk.com/', 'https://www.avenueclinic.co.kr/', 'https://drbong.modoo.at/', 'https://youngclinic.imweb.me/', 'https://www.drnoh.kr/', 'http://www.welldental.co.kr/', 'https://kduro.com/']
result_list1 = []
result_list2 = []

if __name__ == '__main__':
    for url in pension_site_list:
        result = script_position_check(url)
        result_list1.extend(result)  # 각 사이트 결과를 result_list에 추가
    
    for url in pension_site_list:
        result = script_position_check(url)
        result_list2.extend(result)  # 각 사이트 결과를 result_list에 추가
    
    # 모든 결과를 하나의 엑셀 파일로 저장
    df1 = pd.DataFrame(result_list1, columns=['Script Content'])
    df2 = pd.DataFrame(result_list2, columns=['Script Content'])
    df1.to_excel(f'{result_list1}output.xlsx', index=False)
    df2.to_excel(f'{result_list2}output.xlsx', index=False)
    print("Results saved to Excel.")
    