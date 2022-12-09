import json
from strategy_report_analyzer.report_types import MT4StrategyReport,TradeTypeReport
import pprint
from fastapi import FastAPI, File, UploadFile
from strategy_report_analyzer.model import MT4StrategyReportModel
import uvicorn
import os
import pymongo
from bson import json_util 

# pprint = pprint.PrettyPrinter(indent=1).pprint

app = FastAPI()

if not os.path.exists('mt4_raw_report'):
    os.makedirs('mt4_raw_report')



@app.post("/MT4Report/strategy")
async def post_mt4_strategy_report(strategy_report_file: UploadFile = File(),strategy_set_file: UploadFile = File()):
    strategy_report_contents,strategy_report_filename = strategy_report_file.file.read(),strategy_report_file.filename.split('.')[0]
    strategy_set_contents,strategy_set_filename = strategy_set_file.file.read(),strategy_set_file.filename.split('.')[0]

    if not strategy_report_file.filename.endswith('.txt'):
        return {'error': 'strategy_report_file type must be txt'}

    if not strategy_set_file.filename.endswith('.set'):
        return {'error': 'strategy_set_file type must be set'}

    if strategy_set_filename != strategy_report_filename:
        return {'error': 'file name must be same'}

    if not os.path.exists(f'mt4_raw_report/{strategy_report_filename}'):
        os.makedirs(f'mt4_raw_report/{strategy_report_filename}')

    with open(f'mt4_raw_report/{strategy_report_filename}/{strategy_report_filename}.txt', 'wb') as f:
        f.write(strategy_report_contents)
    with open(f'mt4_raw_report/{strategy_report_filename}/{strategy_set_filename}.set', 'wb') as f:
        f.write(strategy_set_contents)

    macd_strategy = MT4StrategyReport(name=strategy_report_filename, description='the description of this strategy is empty')
    macd_strategy.read_raw_report_file(f'mt4_raw_report/{strategy_report_filename}/{strategy_report_filename}.txt')
    macd_strategy.gen_profit_loss_list()
    macd_strategy.analyze()
    # macd_strategy.export()

    # # To MongoDB
    db_client = pymongo.MongoClient("mongodb://mongodb:27017/")
    # insert json data into mongodb
    report_db = db_client['report']
    
    strategy_report_col = report_db['strategy_report']

    # strategy_set_col = report_db['strategy_set']
    # print(strategy_set_contents.decode('utf-8'))
    # strategy_set_col.insert_one(strategy_set_contents)
    strategy_report_col.insert_one(macd_strategy.data)


@app.get("/report/strategy")
async def get_strategy_report():
    db_client = pymongo.MongoClient("mongodb://mongodb:27017/")
    db = 'report'
    doc = 'strategy_report'
    mydb = db_client[db]
    mycol = mydb[doc]
    mydoc = mycol.find({})
    data = json.loads(json_util.dumps(list(mydoc)))
    return data


if __name__ == '__main__':

    uvicorn.run(app, host="0.0.0.0", port=7010)

    # StrategyReport Sample
    # macd_strategy = MT4StrategyReport(name='test_report', description='test description')
    # macd_strategy.read_raw_report('mt4/example_report.txt')
    # macd_strategy.gen_profit_loss_list()
    # macd_strategy.analyze()
    # macd_strategy.export()


    # martin = MT4StrategyReport(
    #     name='v_martin', description='優化馬丁策略的進出場條件，並透過市場波動計算區間，對停利停損點位做更好的優化。', resample='D')
    # martin.parse('mt4/martin.txt')
    # martin.export()


    # smartin = MT4StrategyReport(
    #     name='martin_v7', description='高報酬高風險策略，透過馬丁策略搭配市場型態下天地單，進而增加報酬。', resample='D')
    
    # smartin.parse('mt4/smartin.txt')
    # smartin.export()

    # trade_type_name = '外匯交易(FX)'
    # trade_description = '外匯交易透過買入一種貨幣同時賣出另一種貨幣，因此外匯報價總是以貨幣對的形式出現，熱門的貨幣對（EUR/USD）、（USD/JPY）、(GBP/USD）等等。'
    # trade_commidity = ["EUR/USD", "USD/JPY", "GBP/USD", "EUR/JPY","AUD/USD","NZD/USD","USD/CAD","USD/CHF",'AUD/CAD','AUD/JPY']

    # report = TradeTypeReport(trade_type_name=trade_type_name,
    #                         trade_description=trade_description, trade_commidity=trade_commidity)

    # report.add_strategy(martin.data)
    # report.add_strategy(macd_strategy.data)
    # report.add_strategy(smartin.data)

    # report.gen_json()
    # print(report)
