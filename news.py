## -----------------------------------------
## getting news via news api 
## -----------------------------------------

# must pip install newsapi-python first
from newsapi import NewsApiClient 
import pandas as pd

newsapi = NewsApiClient(api_key='f1ed3a48975546cf91abc5c44471a4f7')

sources = newsapi.get_sources(category="general", language="en",country="us") 
filtered_sources = [sources['sources'][x]['id'] for x in range(len(sources['sources']))]
str_sources = ",".join(filtered_sources)
articles = newsapi.get_everything(q='google AND ai AND invest AND revenue',
                                      sources=str_sources,
                                      from_param='2024-03-14',
                                      to='2024-04-14',
                                      language='en',
                                      sort_by='relevancy')

filtered_article_titles = [articles['articles'][x]['title'] for x in range(len(articles['articles']))]
authors = [articles['articles'][x]['author'] for x in range(len(articles['articles']))]
descriptions = [articles['articles'][x]['description'] for x in range(len(articles['articles']))]
urls = [articles['articles'][x]['url'] for x in range(len(articles['articles']))]
dates = [articles['articles'][x]['publishedAt'] for x in range(len(articles['articles']))]
dict = {'title':filtered_article_titles, 'author':authors, 'description':descriptions, 'url':urls, 'date':dates}
df = pd.DataFrame(dict) 
# df.to_csv('out.csv')

## -----------------------------------------
## using pretrained bert to make predictions 
## -----------------------------------------

# pip install torch, transformers
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Set model to evaluation mode
model.eval()

# Define the input text
annotated = pd.read_csv('annotated.csv')
titles = annotated['title'].tolist()
descriptions = annotated['description'].tolist()
ratings = (annotated['rating']*2).tolist()

mse_titles = 0
acc_titles = 0
mse_descriptions = 0
acc_descriptions = 0

text_data = [titles, descriptions]
sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
count = 0
for data in text_data:
  
  for i, text in enumerate(data): 
  # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

    # Perform forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted probabilities
    probs = softmax(outputs.logits, dim=1)

    # Get predicted label
    predicted_label = torch.argmax(probs, dim=1).item()

    # count = 0 for titles, 1 for descriptions
    acc_titles += 1 if predicted_label == ratings[i] and count == 0 else 0
    acc_descriptions += 1 if predicted_label == ratings[i] and count == 1 else 0

    mse_titles += (probs-ratings[i])**2 if predicted_label == ratings[i] and count == 0 else 0
    mse_descriptions += (probs-ratings[i])**2 if predicted_label == ratings[i] and count == 1 else 0

    # Map predicted label to sentiment
    predicted_sentiment = sentiment_mapping[predicted_label]

    # Print results
    print(f"Text: {text}")
    print(f"Predicted sentiment: {predicted_sentiment}")
    print(f"Predicted probabilities: {probs.tolist()[0]}")
  count += 1

print("\naccuracy from titles: " + str(acc_titles/len(titles)))
print("accuracy from descriptions: " + str(acc_descriptions/len(titles)))
print("\mse from titles: " + str(mse_titles/len(titles)))
print("\mse from descriptions: " + str( mse_descriptions/len(titles)))
