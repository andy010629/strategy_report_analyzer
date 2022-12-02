from report_types import MT4StrategyReport,TradeTypeReport




if __name__ == '__main__':


    macd_strategy = MT4StrategyReport(
        name='macd_std', description='趨勢策略，觀察市場波動及標準差，並搭配 triple_ema 及 macd ，進而優化進出場條件。', resample='D')
    macd_strategy.parse('mt4/macd.txt')
    macd_strategy.export()


    martin = MT4StrategyReport(
        name='v_martin', description='優化馬丁策略的進出場條件，並透過市場波動計算區間，對停利停損點位做更好的優化。', resample='D')
    martin.parse('mt4/martin.txt')
    martin.export()


    smartin = MT4StrategyReport(
        name='martin_v7', description='高報酬高風險策略，透過馬丁策略搭配市場型態下天地單，進而增加報酬。', resample='D')
    
    smartin.parse('mt4/smartin.txt')
    smartin.export()

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
