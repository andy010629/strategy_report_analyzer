import json
import pymongo
from strategy_report_analyzer.report_types import MT4StrategyReport,TradeTypeReport
import pprint


pprint = pprint.PrettyPrinter(indent=1).pprint

def dataframe_to_mongo(db,doc,select_df):
    mydb = db_client[db]
    mycol = mydb[doc]
    records = select_df.to_dict('records')
    mycol.insert_many(records)

if __name__ == '__main__':


  # StrategyReport Sample
  macd_strategy = MT4StrategyReport(name='test_report', description='test description')
  macd_strategy.read_raw_report_file('mt4/example_report.txt')
  macd_strategy.gen_profit_loss_list()
  macd_strategy.analyze()
  macd_strategy.export()

  # # TradeTypeReport Sample
  # trade_type_name = '外匯交易(FX)'
  # trade_description = '外匯交易透過買入一種貨幣同時賣出另一種貨幣，因此外匯報價總是以貨幣對的形式出現，熱門的貨幣對（EUR/USD）、（USD/JPY）、(GBP/USD）等等。'
  # trade_commidity = ["EUR/USD", "USD/JPY", "GBP/USD", "EUR/JPY","AUD/USD","NZD/USD","USD/CAD","USD/CHF",'AUD/CAD','AUD/JPY']

  # report = TradeTypeReport(trade_type_name,trade_description,trade_commidity)
  # report.add_strategy(macd_strategy.data)
  # report.export()

  # # To MongoDB
  # db_client = pymongo.MongoClient("mongodb://localhost:27017/")
  # db = 'strategy_report'
  # doc = 'trade_type'
  # # insert json data into mongodb
  # mydb = db_client[db]
  # mycol = mydb[doc]
  # mycol.insert_one(json.loads(report.data))


