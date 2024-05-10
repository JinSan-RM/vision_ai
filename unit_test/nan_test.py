import pandas as pd
pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 100)

df = pd.read_excel('script_shoping_result.xlsx')

script_list = df["Script"].to_list()

# print(script_list)
for i, script in enumerate(script_list):
    print(i, " : ", script, type(script))
    print()

script_list = list(filter(None, script_list))
script_list = list(filter(' ', script_list))

print('=================================')
print('=================================')
print('=================================')

for i, script in enumerate(script_list):
    print(i, " : ", script, type(script))
    print()