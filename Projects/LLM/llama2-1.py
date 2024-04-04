import llama2

print("Hello world")

print(dir(llama2))

prompt = """
    What are fun activities I can do this weekend?
"""
response = llama2(prompt)
print(response)