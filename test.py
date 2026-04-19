from frontend import VietnameseTokenizer

texts = ["Xin chào", "ViFlow là một framework TTS"]
tokenizer = VietnameseTokenizer()
token_ids = tokenizer.encode_batch(texts)
print("Token IDs:")
print(token_ids)