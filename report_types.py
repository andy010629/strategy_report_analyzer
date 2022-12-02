from dataclasses import dataclass,  field
from typing import List
import pandas as pd
from tqdm import tqdm
import json
from utils import remove_timezone
import json
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, StrategyReport):
            return obj.json()
        return json.JSONEncoder.default(self, obj)


def get_mdd(df):
    tt,dd = 0,100000000000
    mt,md = 0,0
    mdd = 0
    for row in df.itertuples():

        dd = min(dd, row.margin)
        if row.margin > tt:
            tt = row.margin
            dd = row.margin

        mdd = min(mdd, (dd-tt)/tt)
        # if (dd-tt)/tt < mdd:
        #     mdd = (dd-tt)/tt
        #     mt = tt
        #     md = dd

    return mdd*100

def get_max_continue_loss(df):
    mcl, cl = 0, 0

    for row in df.itertuples():
        if row.profit_2 > 0: cl = 0
        else: cl += row.profit_2 
        mcl = min(mcl, cl)
    return mcl 

def get_max_continue_profit(df):
    mcp, cp = 0, 0
    for row in df.itertuples():
        if row.profit_2 < 0: cp = 0
        else: cp += row.profit_2 
        mcp = max(mcp, cp)
    return mcp


@dataclass
class StrategyReport:
    name: str 
    description: str 
    weight: float = 0
    profit_loss_list: List = field(default_factory=list)
    resample:str = 'D'

    win_times = 0
    lose_times = 0
    trade_times = 0
    win_rate = 0

    profit_loss: float = 0
    last_month_profit_loss : float = 0
    last_year_profit_loss: float = 0

    max_profit: float = 0
    max_loss: float = 0
    mdd: float = 0
    max_continue_profit: float = 0
    max_continue_loss: float = 0

    def json(self):
        self.name = self.name.replace(
            ' ', '_').replace('.', '_').replace('/', '_')

        return {
            'name': self.name,
            'description': self.description,
            'weight': self.weight,
    
            'profit_loss': self.profit_loss,
            'last_month_profit_loss': self.last_month_profit_loss,
            'last_year_profit_loss': self.last_year_profit_loss,
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
    def data(self):
        return self.json()

    def export(self):
        self.name = self.name.replace(
            ' ', '_').replace('.', '_').replace('/', '_')

        with open(f'strategy_report/{self.name}.json', 'w') as f:
            json.dump(self.json(), f, cls=NpEncoder)

class MT4StrategyReport(StrategyReport):

    def parse(self, filepath: str, plot=False) -> dict:

        trade_list = []
        with open(filepath, 'r') as f:
            for line in tqdm(f.readlines()):
                line = line.strip()
                order_id, time, op, order_size, qty, price, stop_loss, profit_1, profit_2, margin = line.split(
                    '\t')
                datetime = pd.to_datetime(time)
                tmpdata = {
                    'order_id': order_id,
                    'datetime': datetime,
                    'op': op,
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
        df['margin'] = df['margin'].astype(float)
        df['profit_2'] = df['profit_2'].astype(float)
        df['profit_loss'] = df['margin']
        df['profit_loss_percent'] = df['profit_loss']/df.profit_loss.iloc[0]*100




        last_year_df = df[df['datetime'].dt.year == df.iloc[-1]['datetime'].year]
        last_month_df = last_year_df[last_year_df['datetime'].dt.month == last_year_df.iloc[-1]['datetime'].month]
        # self.trade_times = len(df)
        self.profit_loss = round((df['profit_loss_percent'].iloc[-1] - df['profit_loss_percent'].iloc[0]), 2)

        # self.last_month_profit_loss = round(
        #     (1-(100+last_month_df['profit_loss_percent'].iloc[0])/(100+last_month_df['profit_loss_percent'].iloc[-1])), 2)*100
        
        # self.last_year_profit_loss = round(
        #     (1-(100+last_year_df['profit_loss_percent'].iloc[0])/(100+last_year_df['profit_loss_percent'].iloc[-1])), 2)*100

        self.last_month_profit_loss = round((last_month_df['profit_loss_percent'].iloc[-1] -
               last_month_df['profit_loss_percent'].iloc[0])/last_month_df['profit_loss_percent'].iloc[0],2)*100

        self.last_year_profit_loss = round((last_year_df['profit_loss_percent'].iloc[-1] -
               last_year_df['profit_loss_percent'].iloc[0])/last_year_df['profit_loss_percent'].iloc[0],2)*100

        # print(last_year_df)

        # print(df['profit_loss_percent'])
        # print(self.profit_loss)
        # print(self.last_year_profit_loss)


        # times
        for p in df['profit_2']:
            if float(p) > 0: self.win_times += 1
            if float(p) < 0: self.lose_times += 1
        self.trade_times = self.win_times + self.lose_times
        self.win_rate = self.win_times / self.trade_times * 100

        # max profit/loss
        self.max_profit = df['profit_2'] .max()
        self.max_loss = df['profit_2'] .min()
        self.mdd = get_mdd(df)
        self.max_continue_profit = get_max_continue_profit(df)
        self.max_continue_loss = get_max_continue_loss(df)

        # clear data to dict
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        df = df.resample(self.resample).last().ffill()
        df['datetime'] = df.index
        df['datetime'] = df['datetime'].apply(remove_timezone)
        df['datetime'] = df['datetime'].astype(str)
        profit_loss_data = df[['datetime', 'profit_loss_percent']].to_dict('records')
        # print(profit_loss_data)
        self.profit_loss_list = profit_loss_data

        if plot:
            sns.lineplot(x='datetime', y='profit_loss', data=df)
            plt.show()

        return profit_loss_data

@dataclass
class TradeTypeReport:
    trade_type_name: str
    trade_description: str
    trade_commidity: field(default_factory=list)
    strategy_list: List[StrategyReport] = field(default_factory=list)

    def add_strategy(self, strategy, weight=1,full_profit_loss=False):
        if not full_profit_loss:
            strategy['profit_loss_list'] = strategy['profit_loss_list'][-100:]
        strategy['weight'] = weight
        self.strategy_list.append(strategy)

    def add_commidity(self, commidity):
        self.trade_commidity.append(commidity)

    def gen_json(self):
        data = {}
        trade_type_name = self.trade_type_name.replace(
            ' ', '_').replace('.', '_').replace('/', '_')
        data['trade_type_name'] = trade_type_name
        data['trade_description'] = self.trade_description
        data['trade_commidity'] = self.trade_commidity
        data['strategy_list'] = self.strategy_list
        with open(f'trade_report/{trade_type_name}.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, cls=MyEncoder, ensure_ascii=False))
