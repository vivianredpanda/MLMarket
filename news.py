# must pip install newsapi-python first
from newsapi import NewsApiClient 

import json 

# Init
newsapi = NewsApiClient(api_key='f1ed3a48975546cf91abc5c44471a4f7')

sources = newsapi.get_sources(category="general", language="en",country="us") 
filtered_sources = [sources['sources'][x]['id'] for x in range(len(sources['sources']))]
str_sources = ",".join(filtered_sources)
articles = newsapi.get_everything(q='ai',
                                      sources=str_sources,
                                      from_param='2024-03-12',
                                      to='2024-03-20',
                                      language='en',
                                      sort_by='relevancy')
filtered_article_titles = [articles['articles'][x]['title'] for x in range(len(articles['articles']))]