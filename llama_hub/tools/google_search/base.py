"""Google Search tool spec."""

from llama_index.tools.tool_spec.base import BaseToolSpec
from llama_index.readers.schema.base import Document
import requests
import json
import urllib.parse
from typing import Optional
from bs4 import BeautifulSoup


QUERY_URL_TMPL = (
    "https://www.googleapis.com/customsearch/v1?key={key}&cx={engine}&q={query}&{siteSearch}"
)


class GoogleSearchToolSpec(BaseToolSpec):
    """Google Search tool spec."""

    spec_functions = ["google_search"]
    global result_source
    def __init__(self, key: str, engine: str, siteSearch: str,num: Optional[int] = None) -> None:
        """Initialize with parameters."""
        self.key = key
        self.engine = engine
        self.num = num
        self.siteSearch=siteSearch
        print(siteSearch)

    def google_search(self, query: str):
        """
        Make a query to the Google search engine to receive a list of results.

        Args:
            query (str): The query to be passed to Google search.
            num (int, optional): The number of search results to return. Defaults to None.

        Raises:
            ValueError: If the 'num' is not an integer between 1 and 10.
        """
        url = QUERY_URL_TMPL.format(
                key=self.key,
                engine=self.engine,
                query=urllib.parse.quote_plus(query),
                siteSearch=self.siteSearch
        )
        if self.num is not None:
            if not 1 <= self.num <= 10:
                raise ValueError("num should be an integer between 1 and 10, inclusive")
            url += f"&num={self.num}"
        print(url)
        response = requests.get(url)
      
        return [Document(text=response.text)]
    
    def get_source_url(self, query: str):
        url = QUERY_URL_TMPL.format(
                key=self.key,
                engine=self.engine,
                query=urllib.parse.quote_plus(query),
                siteSearch=self.siteSearch
        )
        response = requests.get(url)
        
        result_source=json.loads(response.text).get('items', [])
        return result_source