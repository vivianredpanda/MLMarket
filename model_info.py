from pytrends.request import TrendReq

pytrends = TrendReq(hl="en-US", tz=360)

suggestions_c = pytrends.suggestions("chatgpt")
chatgpt = [d for d in suggestions_c if d["type"] == "Software"]
chatgpt = chatgpt[0]["mid"]
suggestions_g = pytrends.suggestions("gemini")
gemini = [d for d in suggestions_g if d["type"] == "Chatbot"]
gemini = gemini[0]["mid"]
print(chatgpt, gemini)

pytrends.build_payload(kw_list=[gemini], timeframe=["2023-06-05 2025-03-17"])
chatgpt_data = pytrends.interest_over_time()
chatgpt_data.to_csv("correlation analysis/gemini_interest.csv")

pytrends.build_payload(kw_list=[chatgpt], timeframe=["2023-06-05 2025-03-17"])
gemini_data = pytrends.interest_over_time()
gemini_data.to_csv("correlation analysis/chatgpt_interest.csv")
