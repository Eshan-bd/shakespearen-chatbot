import pandas as pd
import json
import os

# Load Hamlet utterances
file_path='data/raw/hamlet_utterances.jsonl'
df_hamletUt=pd.read_json(file_path, lines=True)

# Clean text and remove unwanted rows
df_hamletUt['text'] = df_hamletUt['text'].str.replace(r'\[.*?\]', '', regex=True)
df_hamletUt['text'] = df_hamletUt['text'].str.replace(r'[\[\]]', '', regex=True)
df_hamletUt = df_hamletUt[df_hamletUt['speaker'] != 'STAGE_DIRECTION']
df_hamletUt['text'] = df_hamletUt['text'].str.strip()
df_hamletUt = df_hamletUt[(df_hamletUt['act'] != 0) & (df_hamletUt['scene'] != 0)]

print(df_hamletUt.isnull().sum())

# Save cleaned Hamlet utterances
os.makedirs('data/processed', exist_ok=True)
df_hamletUt.to_json('data/processed/hamlet_ut.jsonl', orient='records', lines=True)
df_hamletUt.to_json('data/processed/hamlet.json', orient='records', indent=2)

# Load Macbeth utterances
file_path='data/raw/macbeth_utterances.jsonl'
df_mcbethUt=pd.read_json(file_path, lines=True)

# Clean text and remove unwanted rows
df_mcbethUt['text'] = df_mcbethUt['text'].str.replace(r'\[.*?\]', '', regex=True)
df_mcbethUt['text'] = df_mcbethUt['text'].str.replace(r'[\[\]]', '', regex=True)
df_mcbethUt = df_mcbethUt[df_mcbethUt['speaker'] != 'STAGE_DIRECTION']
df_mcbethUt['text'] = df_mcbethUt['text'].str.strip()
df_mcbethUt = df_mcbethUt[(df_mcbethUt['act'] != 0) & (df_mcbethUt['scene'] != 0)]

print(df_mcbethUt.isnull().sum())

# Save cleaned Macbeth utterances
df_mcbethUt.to_json('data/processed/macbeth_ut.jsonl', orient='records', lines=True)
df_mcbethUt.to_json('data/processed/macbeth.json', orient='records', indent=2)

# Load Romeo and Juliet utterances
file_path='data/raw/romeo_and_juliet_utterances.jsonl'
df_romeo_julietUt=pd.read_json(file_path, lines=True)

# Clean text and remove unwanted rows
df_romeo_julietUt['text'] = df_romeo_julietUt['text'].str.replace(r'\[.*?\]', '', regex=True)
df_romeo_julietUt['text'] = df_romeo_julietUt['text'].str.replace(r'[\[\]]', '', regex=True)
df_romeo_julietUt = df_romeo_julietUt[df_romeo_julietUt['speaker'] != 'STAGE_DIRECTION']
df_romeo_julietUt['text'] = df_romeo_julietUt['text'].str.strip()
df_romeo_julietUt = df_romeo_julietUt[(df_romeo_julietUt['act'] != 0) & (df_romeo_julietUt['scene'] != 0)]

# Save cleaned Romeo and Juliet utterances
df_romeo_julietUt.to_json('data/processed/romeo_juliet_ut.jsonl', orient='records', lines=True)
df_romeo_julietUt.to_json('data/processed/romeo_and_juliet.json', orient='records', indent=2)

# Print chunk size stats
print("Hamlet utterances:", len(df_hamletUt))
print("Macbeth utterances:", len(df_mcbethUt))
print("Romeo and Juliet utterances:", len(df_romeo_julietUt))

# Flag chunks that are too long for retrieval
df_hamletUt['too_long'] = df_hamletUt['text'].str.len() > 2000
df_mcbethUt['too_long'] = df_mcbethUt['text'].str.len() > 2000
df_romeo_julietUt['too_long'] = df_romeo_julietUt['text'].str.len() > 2000

print("Hamlet too long:", df_hamletUt['too_long'].value_counts())
print("Macbeth too long:", df_mcbethUt['too_long'].value_counts())
print("Romeo and Juliet too long:", df_romeo_julietUt['too_long'].value_counts())
