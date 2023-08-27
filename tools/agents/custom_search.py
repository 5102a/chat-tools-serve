import json
import requests

GOOGLE_CSE_ID = "e7f1a28840e8c4715"
GOOGLE_API_KEY = "AIzaSyBA7QfcsE2NnhqmIFUTiuYadZfHKWeF3lk"
RapidAPIKey = "90bbe925ebmsh1c015166fc5e12cp14c503jsn6cca55551ae4"


class DeepSearch:
    def search(query: str = ""):
        query = query.strip()
        print(query)
        if query == "":
            return ""

        if RapidAPIKey == "":
            return "请配置你的 RapidAPIKey"

        url = "https://bing-web-search1.p.rapidapi.com/search"

        querystring = {"q": query,
                       "mkt": "zh-cn", "textDecorations": "false", "setLang": "CN", "safeSearch": "Off", "textFormat": "Raw"}

        headers = {
            "Accept": "application/json",
            "X-BingApis-SDK": "true",
            "X-RapidAPI-Key": RapidAPIKey,
            "X-RapidAPI-Host": "bing-web-search1.p.rapidapi.com"
        }

        response = requests.get(url, headers=headers, params=querystring)
        print(f"search response:{response}")
        data_list = response.json()['value']

        if len(data_list) == 0:
            return ""
        else:
            result_arr = []
            result_str = ""
            count_index = 0
            for i in range(10):
                item = data_list[i]
                title = item["name"]
                description = item["description"]
                item_str = f"{title}: {description}"
                result_arr = result_arr + [item_str]

            result_str = "\n".join(result_arr)
            return result_str

    def google_search(query: str = ""):
        query = query.strip()
        json_res = json.loads(query)
        query = json_res.get('query')
        if query == "":
            return ""

        url = "https://www.googleapis.com/customsearch/v1"

        querystring = {"q": query,
                       "key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "gl": "cn", "hl": "zh-CN"}

        headers = {
            "Accept": "application/json",
        }

        response = requests.get(url, headers=headers, params=querystring)
        res = response.json()
        data_list = res["items"][::-1]

        if len(data_list) == 0:
            return ""
        else:
            result_arr = []
            result_str = ""
            count_index = 0
            for item in data_list:
                title = item["title"]
                description = item["snippet"]
                item_str = f"{title}: {description}"
                result_arr = result_arr + [item_str]
                count_index += 1
                if count_index >= 3:
                    break
            result_str = " ".join(result_arr)
            return result_str
