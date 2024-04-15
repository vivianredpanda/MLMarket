from pytrends.request import TrendReq

pytrends = TrendReq(hl="en-US", tz=360)

suggestions = pytrends.suggestions("chatgpt")
chatgpt = [d for d in suggestions if d["type"] == "Software"]
chatgpt = chatgpt[0]["mid"]
suggestions = pytrends.suggestions("gemini")
gemini = [d for d in suggestions if d["type"] == "Chatbot"]
gemini = gemini[0]["mid"]
# print(chatgpt, gemini)

pytrends.build_payload(kw_list=[gemini], timeframe=["2022-09-04 2024-04-14"])
chatgpt_data = pytrends.interest_over_time()
chatgpt_data.to_csv("chatgpt_interest.csv")

pytrends.build_payload(kw_list=[chatgpt], timeframe=["2022-09-04 2024-04-14"])
gemini_data = pytrends.interest_over_time()
gemini_data.to_csv("gemini_interest.csv")
