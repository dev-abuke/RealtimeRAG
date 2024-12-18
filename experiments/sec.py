import logging
from datetime import timedelta
import re, sys
import xml.etree.ElementTree as ET
import json
from typing import Dict
from loguru import logger

from pandas import read_json
import requests
from bytewax import operators as op
from bytewax.connectors.files import FileSink
from bytewax.dataflow import Dataflow
from bytewax.inputs import FixedPartitionedSource, SimplePollingSource
# from bytewax.connectors.kafka import operators as kop
# from bytewax.connectors.kafka import KafkaSinkMessage
logger.add(sys.stderr, format="{time} {level} {message} {line}", filter="sec_filing", level="INFO")

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

class SECSource(SimplePollingSource):
    def next_item(self):
        # Base URL for SEC Edgar
        base_url = "https://www.sec.gov/cgi-bin/browse-edgar"

        # User agent header to mimic a browser (SEC requires this to allow access)
        headers = {
            'User-Agent': 'Bytewax, Inc. contact@bytewax.io',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.5',
            'Cache-Control':'no-cache',
            'Host': 'www.sec.gov'
        }

        # https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&CIK=0000070858&type=&company=&dateb=&owner=include&start=0&count=40&output=atom
        params = {
            'action':'getcurrent',
            'CIK': '',
            'type':'',
            'dateb':'',
            'owner':'include',
            'start':'0',
            'count':'200',
            'output':'atom'    # Number of results to return
        }

        # Making the GET request
        response = requests.get(base_url, headers=headers, params=params)
        
        if response.status_code == 200:
            logger.info(f"Successfully retrieved filings :: {response.text[:300]}")
            return response.text
        else:
            logger.info(f"Failed to retrieve filings. Status code: {response.status_code}")
            return None


flow = Dataflow("edgar_scraper")
filings_stream = op.input("in", flow, SECSource(timedelta(seconds=10)))


def parse_atom(xml_data):
    logger.info(f"Parsing XML data... {xml_data[:300]}")
    # Parse the XML data
    # Set up the namespace map
    namespace = {"atom": "http://www.w3.org/2005/Atom"}

    # Parse the XML
    root = ET.fromstring(xml_data)
    data = []
    # Iterate over each entry and extract the desired information
    for entry in root.findall("atom:entry", namespace):
        logger.info(f"Processing entry... {entry}")
        id = entry.find("atom:id", namespace).text.split("=")[-1].replace("-", "")
        title = entry.find("atom:title", namespace).text
        link = entry.find("atom:link[@type='text/html']", namespace).get("href")
        cik_match = re.search(r'\((\d+)\)', title)
        cik = cik_match.group(1) if cik_match else "No CIK found"
        form_type = entry.find('atom:category', namespace).attrib['term']

        json_entry =  ("All",
            {
                "id":id,
                "title":title,
                "link":link,
                "cik":cik,
                "form_type":form_type

            })

        data.append(
           json_entry
        )

        if len(data) == 10:
            return data

    logger.info(f" Parsed {len(data)} entries from XML data. First entry: {data[0]}")
    return data

processed_stream = op.flat_map("parse_atom", filings_stream, parse_atom)
op.inspect("processed_stream", processed_stream)

# reduced_parsed_size = op.collect("reduce_item_size", processed_stream, timedelta(seconds=10), 2)
# op.inspect("reduced_parsed_size", reduced_parsed_size)

# reduce_processed_size = op.filter_map("reduce_item_size", filings_stream, lambda x: x[1])

def deduplicate(state, filing):
    logger.info(f"Deduping filings... {state} ###### {filing}")
    if not state:
        state = []
    if filing["id"] in state:
        return (state, None)
    else:
        state.append(filing["id"])
        return (state, filing)

deduped_stream = op.stateful_map("dedupe", processed_stream, deduplicate)
# op.inspect("dedupe_stream", deduped_stream)

deduped_filtered_stream = op.filter_map("remove key", deduped_stream, lambda x: x[1])
# op.inspect("filt", deduped_filtered_stream)

# Comapany tickers schema
# {"0":{"cik_str":789019,"ticker":"MSFT","title":"MICROSOFT CORP"}, ...}
cik_to_ticker = read_json("company_tickers.json", orient='index')
cik_to_ticker.set_index(cik_to_ticker['cik_str'], inplace=True)

def enrich(data, cik_to_tickers):
    cik = int(data["cik"])
    try:
        ticker = cik_to_ticker['ticker'].loc[cik]
        if not isinstance(ticker, str):
            ticker.iloc[0]
    except KeyError:
        logger.critical("no valid ticker, checking form")
        if str(data["form_type"]) in ["5", "3", "4"]:

            # Split the URL into parts
            parts = data['link'].split('/')

            # Extract the relevant parts
            cik = parts[6]
            accession_number = parts[7]
            formatted_accession_number = parts[-1].replace('-index.htm', '.txt')

            # Construct the new URL
            new_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{formatted_accession_number}"

            # User agent header to mimic a browser (SEC requires this to allow access)
            headers = {
                'User-Agent': 'Bytewax, Inc. contact@bytewax.io',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Accept-Language': 'en-US,en;q=0.5',
                'Cache-Control':'no-cache',
                'Host': 'www.sec.gov'
            }

            response = requests.get(new_url, headers=headers)
        
            if response.status_code == 200:
                logger.success("Successfully retrieved filing text")
                pattern = re.compile(r'<issuerTradingSymbol>(.*?)</issuerTradingSymbol>', re.DOTALL)
                ticker = pattern.findall(response.text)
                if ticker == []:
                    logger.error(f"Failed to retrieve ticker from {response.text}")
                    return None
                return (ticker[0], data)
            else:
                logger.error(f"Failed to retrieve filings. Status code: {response.status_code}")
                return None
        
        logger.warning("no valid ticker and wrong type")
        # return {"ticker": None, **data}
        return ("no_ticker", data)
    # return {"ticker": ticker, **data}
    return (ticker, data)

enrich_stream = op.filter_map("enrich", deduped_filtered_stream, lambda x: enrich(x, cik_to_ticker))
# op.inspect("enriched_streram", enrich_stream)

# def serialize_k(news)-> KafkaSinkMessage[Dict, Dict]:
#     return KafkaSinkMessage(
#         key=json.dumps(news['symbols'][0]),
#         value=json.dumps(news),
#     )

def serialize(data):
    try:
        return ('All', json.dumps(data))
    except:
        return ('All', json.dumps(''))

serialized = op.map("serialize", enrich_stream, serialize)
op.output("output", serialized, FileSink('sec_out2.jsonl'))

## uncomment to write to kafka
# serialized = op.map("serialize", enrich_stream, serialize_k)
# kop.output("out", serialized, brokers=BROKERS, topic=OUT_TOPIC)