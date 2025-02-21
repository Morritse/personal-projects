#!/bin/bash
curl --request GET 'https://paper-api.alpaca.markets/v2/assets?asset_class=crypto' \
--header 'Apca-Api-Key-Id: PKBG81XG90DTMLE83LQ7' \
--header 'Apca-Api-Secret-Key: hQgygItKU6YD66ZyBtCRuDSyo9YGesuYJOpyJsG3' | \
python3 -c '
import sys, json
data = json.load(sys.stdin)
tradable = [asset for asset in data if asset["tradable"]]
print("\nTradable Crypto Pairs:")
print("=====================")
for asset in tradable:
    print(f"{asset["symbol"]:<10} Min Order: {asset["min_order_size"]}")
'
