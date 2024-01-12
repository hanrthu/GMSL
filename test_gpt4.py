import openai, os
key = "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"
os.environ["OPENAI_API_KEY"] = key
openai.api_key = os.getenv("OPENAI_API_KEY")
prompt = "which version of gpt you are."
completion = openai.ChatCompletion.create(
    model="gpt-4-1106-preview",
    messages=[
        {"role": "user", "content":prompt},
    ],
    temperature=0
)
print(completion.choices[0].message["content"])