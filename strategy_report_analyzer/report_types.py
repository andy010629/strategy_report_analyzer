from dataclasses import dataclass,  field
from typing import List
import pandas as pd
from tqdm import tqdm
import json
import json
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from strategy_report_analyzer.utils import remove_timezone
import matplotlib.ticker as mtick
import os

if not os.path.exists('report/strategy'):
    os.makedirs('report/strategy')

if not os.path.exists('report/tradetype'):
    os.makedirs('report/tradetype')


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
class StrategyReportEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, StrategyReport):
            return obj.json()
        return json.JSONEncoder.default(self, obj)

@dataclass
class StrategyReport:
    name: str 
    description: str = ""
    weight: float = None
    profit_loss_list: List = field(default_factory=list)
    win_times:int = None
    lose_times:int = None
    trade_times:int = None
    win_rate:float = None
    profit_loss_percent: float = None
    last_month_profit_loss : float = None
    last_year_profit_loss_percent: float = None
    max_profit: float = None
    max_loss: float = None
    mdd: float = None
    max_continue_profit: float = None
    max_continue_loss: float = None


    @staticmethod
    def get_mdd(df:pd.DataFrame) -> float:
        tt,dd = 0,100000000000
        mdd = 0
        for row in df.itertuples():
            dd = min(dd, row.margin)
            if row.margin > tt:
                tt = row.margin
                dd = row.margin

            mdd = min(mdd, (dd-tt)/tt)
        return round(mdd*100,2)

    @staticmethod
    def get_max_continue_loss(df:pd.DataFrame) -> float:
        mcl, cl = 0, 0

        for row in df.itertuples():
            if row.profit_2 > 0: cl = 0
            else: cl += row.profit_2 
            mcl = min(mcl, cl)
        return round(mcl,2) 

    @staticmethod
    def get_max_continue_profit(df:pd.DataFrame):
        mcp, cp = 0, 0
        for row in df.itertuples():
            if row.profit_2 < 0: cp = 0
            else: cp += row.profit_2 
            mcp = max(mcp, cp)
        return round(mcp,2)

    def json(self) -> dict:
        self.name = self.name.replace(
            ' ', '_').replace('.', '_').replace('/', '_')

        return {
            'name': self.name,
            'description': self.description,
            'weight': self.weight,
            'profit_loss': self.profit_loss_percent,
            'last_month_profit_loss': self.last_month_profit_loss_percent,
            'last_year_profit_loss': self.last_year_profit_loss_percent,
            'win_times': self.win_times,
            'lose_times': self.lose_times,
            'trade_times': self.trade_times,
            'win_rate': self.win_rate,
            'max_profit': self.max_profit,
            'max_loss': self.max_loss,
            'mdd': self.mdd,
            'max_continue_profit': self.max_continue_profit,
            'max_continue_loss': self.max_continue_loss,
            'profit_loss_list': self.profit_loss_list,
        }

    @property
    def data(self) -> dict:
        return self.json()

    def export(self,folder_path:str='report/strategy')->None:
        self.name = self.name.replace(' ', '_').replace('.', '_').replace('/', '_')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(f'{folder_path}/{self.name}.json', 'w') as f:
            json.dump(self.json(), f, cls=NpEncoder)

class MT4StrategyReport(StrategyReport):
    raw_report: pd.DataFrame = field(init=False)

    def gen_profit_loss_list(self,plot:bool=False,resample:str='') -> list:
        df = self.raw_report.copy()

        # clear data to dict
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        if resample:
            df = df.resample(resample).last().ffill()
        df['datetime'] = df.index
        df['datetime'] = df['datetime'].apply(remove_timezone)
        df['datetime'] = df['datetime'].astype(str)

        self.profit_loss_list = df[['datetime', 'profit_loss_percent']].to_dict('records')
       
        if plot:
            df['datetime'] = pd.to_datetime(df['datetime'], format = '%Y-%m-%d')
            fig = sns.lineplot(x='datetime', y='profit_loss_percent', data=df)
            fig.yaxis.set_major_formatter(mtick.PercentFormatter())
            plt.show()
        
        return self.profit_loss_list

    def read_raw_report_file(self, filepath: str) -> pd.DataFrame:
        trade_list = []
        with open(filepath, 'r') as f:
            for line in tqdm(f.readlines()):
                line = line.strip()
                order_id, time, operation, order_size, qty, price, stop_loss, profit_1, profit_2, margin = line.split('\t')
                datetime = pd.to_datetime(time)
                tmpdata = {
                    'order_id': order_id,
                    'datetime': datetime,
                    'operation': operation,
                    'order_size': order_size,
                    'qty': qty,
                    'price': price,
                    'stop_loss': stop_loss,
                    'profit_1': profit_1,
                    'profit_2': profit_2,
                    'margin': margin
                }
                trade_list.append(tmpdata)

 

        # make df and set type
        df = pd.DataFrame(trade_list)
        df['order_id'] = df['order_id'].astype(int)
        df['operation'] = df['operation'].astype(object)
        df['order_size'] = df['order_size'].astype(float)
        df['qty'] = df['qty'].astype(float)
        df['price'] = df['price'].astype(float)
        df['stop_loss'] = df['stop_loss'].astype(float)
        df['profit_1'] = df['profit_1'].astype(float)
        df['profit_2'] = df['profit_2'].astype(float)
        df['margin'] = df['margin'].astype(float)
        df['profit_2'] = df['profit_2'].astype(float)
    
        # set profit loss 
        df['profit_loss']= df['margin']
        df['profit_loss_percent'] = round(df['profit_loss']/df.profit_loss.iloc[0]*100,2)

        self.raw_report = df
        return self.raw_report

    def analyze(self):
        if self.raw_report is None:
            print('Please read raw report first')
            return
        
        # if self.profit_loss_list is None:
        #     self.profit_loss_list = self.get_profit_loss_list()

        df = self.raw_report.copy()

        last_year_df = df[df['datetime'].dt.year == df.iloc[-1]['datetime'].year]
        last_month_df = last_year_df[last_year_df['datetime'].dt.month == last_year_df.iloc[-1]['datetime'].month]

        # profit_loss
        self.profit_loss_percent = round((df['profit_loss_percent'].iloc[-1] - df['profit_loss_percent'].iloc[0]), 2)
        self.last_month_profit_loss_percent = round((last_month_df['profit_loss_percent'].iloc[-1] -
               last_month_df['profit_loss_percent'].iloc[0])/last_month_df['profit_loss_percent'].iloc[0],2)*100

        self.last_year_profit_loss_percent = round((last_year_df['profit_loss_percent'].iloc[-1] -
               last_year_df['profit_loss_percent'].iloc[0])/last_year_df['profit_loss_percent'].iloc[0],2)*100

        # times
        self.win_times = 0
        self.lose_times = 0
        for p in df['profit_2']:
            if float(p) > 0: self.win_times += 1
            if float(p) < 0: self.lose_times += 1
        self.trade_times = self.win_times + self.lose_times
        self.win_rate = round(self.win_times / self.trade_times * 100,2)

        # max profit/loss
        self.max_profit = round(df['profit_2'] .max(),2)
        self.max_loss = round(df['profit_2'] .min(),2)
        self.mdd = self.get_mdd(df)
        self.max_continue_profit = self.get_max_continue_profit(df)
        self.max_continue_loss = self.get_max_continue_loss(df)


@dataclass
class TradeTypeReport:
    trade_type_name: str
    trade_description: str 
    trade_commidity: field(default_factory=list)
    strategy_list: List[StrategyReport] = field(default_factory=list)

    def add_strategy(self, strategy:StrategyReport, weight:float=1,full_profit_loss:bool=False) -> None:
        if not full_profit_loss:
            strategy['profit_loss_list'] = strategy['profit_loss_list'][-100:]
        strategy['weight'] = weight
        self.strategy_list.append(strategy)

    def add_commidity(self, commidity:str) -> None:
        self.trade_commidity.append(commidity)

    def export(self,folder_path:str='report/tradetype') -> None:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        data = {}
        trade_type_name = self.trade_type_name.replace(' ', '_').replace('.', '_').replace('/', '_')
        data['trade_type_name'] = trade_type_name
        data['trade_description'] = self.trade_description
        data['trade_commidity'] = self.trade_commidity
        data['strategy_list'] = self.strategy_list
        with open(f'{folder_path}/{trade_type_name}.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, cls=StrategyReportEncoder, ensure_ascii=False))

    @property
    def data(self):
        return self.json()

    def json(self):
        data = {}
        trade_type_name = self.trade_type_name.replace(' ', '_').replace('.', '_').replace('/', '_')
        data['trade_type_name'] = trade_type_name
        data['trade_description'] = self.trade_description
        data['trade_commidity'] = self.trade_commidity
        data['strategy_list'] = self.strategy_list
        return json.dumps(data, cls=StrategyReportEncoder, ensure_ascii=False)