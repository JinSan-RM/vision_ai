from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import base64
import time
import openai
import pandas as pd
import os.path


def make_dataframe_to_csv(col_1 : list, col_2 : list):
    csv_file_name = './script_pension_result.csv'
    
    if os.path.isfile("./" + csv_file_name):
        previos_df = pd.read_csv("./" + csv_file_name, index_col=False)
        new_df = pd.DataFrame({"Script" : ["new_script"], "Result_from_GPT" : ["new_gpt_result"]})
        current_df = pd.DataFrame({"Script" : col_1, "Result_from_GPT" : col_2})
        current_df = current_df.replace('\n', '')
        
        modify_current_df = pd.concat([new_df, current_df])
        
        combined_df = pd.concat([previos_df, modify_current_df])
        combined_df.to_csv(csv_file_name, encoding = 'utf-8-sig', index=False)
        
    else:
        df = pd.DataFrame({"Script" : col_1, "Result_from_GPT" : col_2})
        modify_df = df.replace('\n', '')
        
        modify_df.to_csv(csv_file_name, encoding = 'utf-8-sig', index=False)

def script_position_check(url : str):
    print(url)
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
    make_dataframe_to_csv(script_list, word)
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
    token_per_char=0.5
    estimated_tokens = len(data) * token_per_char
    max_tokens = 100 - int(estimated_tokens)
    if max_tokens < 1:
        max_tokens = 1
    if data != '' or data is None:
        word = []
        try:
            for i in data:
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    messages=[
                        {"role": "system", "content": "Your role is a script interpreter. "},
                        {"role": "user", "content": i}
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

if __name__ == '__main__':
        
    sites = ["http://nobleglamping.com/", "http://www.onthebeachpoolvilla.com/", "https://www.pinevallry.co.kr", "http://삐삐키즈풀빌라.com/", "https://jebuilmare.modoo.at"]

    for site in sites:
        result = script_position_check(site)
        print(result)