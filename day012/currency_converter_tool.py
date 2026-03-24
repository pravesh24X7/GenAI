import requests
import os
from langchain_core.tools import tool
from langchain_core.tools import InjectedToolArg
from typing import Annotated
from dotenv import load_dotenv

load_dotenv()

@tool
def get_currency_conversion_rate(input_currency: str, output_currency: str) -> float:
    """
    `get_currency_conversion_rate` function returns the currency rate between input and output currencies.
    
    Input parameters:
    input_currency (string)         : source input currency.
    output_currency(string)         : desired output currency.
    
    Return type:
    rate (float)                    : output currency rate.
    """

    API_KEY = os.environ["EXCHANGE_RATE_API_KEY"]

    URL = f"https://v6.exchangerate-api.com/v6/{API_KEY}/latest/{input_currency}"
    response = requests.get(URL)

    if response.status_code == 200:
        data = response.json()
        rate = data["conversion_rates"][output_currency]
        return rate
    
    return 0.0


@tool
def convert_currency(base_amount: float, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """
    `convert_currency` tool performs the actual conversion of currency on base amount value.
    
    Input parameters:
    base_amount (str)     :   value of money to convert.
    conversion_rate (str) :   rate by which base amount converted.

    Return type:
    result (float)          :   Final result after prforming conversion.
    """
    result = base_amount * conversion_rate
    return result


class CurrencyToolkit:
    def get_tools(self):
        return [get_currency_conversion_rate, convert_currency]
