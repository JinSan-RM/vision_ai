import openai
import pandas as pd
import os.path
import time

### ==============================
# Please Don't forget to input OpenAI API key
### ==============================
# openai.api_key = 

file_name = "script_shoping_result.xlsx"

df = pd.read_excel(file_name, sheet_name=None)

df_sheets = list(df.keys())


def openai_script_analysis(data):
    ##### NOTE : 만약 긴 스크립트들이 의미없다고 생각될 경우 아래 max_tokens 조건 살리기. 
    #            아마 스크립트가 길어서 https://loveparis.net/ 에서 시간이 오래 소요됨
    # token_per_char=0.5
    # estimated_tokens = len(data) * token_per_char
    # max_tokens = 1000 - int(estimated_tokens)
    # if max_tokens < 1:
    #     max_tokens = 1
    if data != '' or data is not None:
        word = []
        try:
            for i in data:
                if i == None:
                    continue
                print(i, type(i), "<=====i")
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
                    # 만약에 코드가 아니라 스크립트 입력이 들어오면, 스크립트가 호출하는 라이브러리의 기능을 알려줘
                    temperature=0.5
                )
                # Access the response data using a method or property
                improved_text = response.choices[0].message.content
                print(improved_text)
                
                word.append(improved_text)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        print(word)
        return word
    else:
        return None
    

# ===========

previos_df = pd.DataFrame()

for i, sheet in enumerate(df_sheets[0:]) :
    
    current_df = df[sheet]
    
    scripts_list = current_df.iloc[0:, 0].to_list()
    new_scripts_list = []
    for script in scripts_list:
        if script == None:
            continue
        new_scripts_list.append(script)
    
    print("<===================== First step start_openai")
    openai_result = openai_script_analysis(new_scripts_list)

    current_df["openai_result_no_js_change"] = openai_result
    print("<===================== First step clear_openai")
    
    
    prepro_scripts_list = []
    for i in new_scripts_list :
        if type(i) != str:
            print(i, type(i), "<=================")
            prepro_scripts_list.append(i)
            continue
        i = i.split(".js")[0] + ".js"
        prepro_scripts_list.append(i)

    current_df["Script_remove_after_js"] = prepro_scripts_list

    print("<===================== Second step start_openai remove after js")
    prepro_openai_result = openai_script_analysis(scripts_list)

    current_df["openai_result_remove_after_js"] = prepro_openai_result
    print("<===================== Second step clear_openai remove after js")   
    
    
    if i == 0 :
        previos_df = current_df
        continue

    previos_df = pd.concat([previos_df, current_df])



print(previos_df)

previos_df.to_excel("test_16.xlsx", index=False)