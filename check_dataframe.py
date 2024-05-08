import pandas as pd

df = pd.read_csv('./script_result.csv', index_col=False)

print(df)

df_mix = pd.DataFrame({'Script':["test"], "Result_from_GPT" : ["test"]})

print(df_mix)
 
concat_df = pd.concat([df, df_mix])

print(concat_df)