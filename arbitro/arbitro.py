import requests
import collections
import time
import itertools
import concurrent.futures 
import random
import threading

from urllib.parse import urlencode
from collections import deque
from functools import reduce
from math import sqrt
from collections import deque, namedtuple


Pair = namedtuple('Pair', ['base', 'quote'])
Op = namedtuple('Op', ['pair', 'op', 'rate'])
Arbitrage = namedtuple('Arbitrage', ['ops', 'weight'])
Trade = namedtuple('Trade', ['op', 'qty'])


def pair_to_name(pair):
    return ('%s-%s' % pair).upper()


class ForecasterServer:

    def __init__(self):
        self.exchange = Bittrex(debug=False)
        self.forecasters = {}

    def run(self):
        for market in self.markets:
            data = self.exchange.market_history(market)
            forecaster_buy = Forecaster()
            forecaster_buy.init(data_buy)
            forecaster_sell = Forecaster()
            forecaster_sell.init(data_sell)
            self.forecasters[(market, 'buy')] = forecaster_buy
            self.forecasters[(market, 'sell')] = forecaster_sell

    def update(self):
        for market in self.markets:
            data = self.exchange.ticker(market)
            self.forecaster[(market, 'buy')].update(data)
            self.forecaster[(market, 'sell')].update(data)

    def serve(self):
        while True:
            market, op = self.conn.recv()
            self.conn.send(self.forecasters[(market, op)].predict())


class Arbitro:
    def __init__(self, curr, exchange):
        #curr = ['btc', 'eth', 'dash', 'bcc', 'btg', 'ltc', 'dft', 'rec', 'zec', 'bcd', 'waves']
        #curr = ['btc', 'bcc', 'eth', 'ltc', 'usdt', 'dash', 'btg'] #, 'ripple', 'ark', 'neo']
        self.exchange = exchange
        pairs = self.exchange.markets()
        pairs = list(filter(lambda p: p.base in curr and p.quote in curr, pairs))
        fwd = [tuple(p) for p in pairs]
        bck = [(p.quote, p.base) for p in  pairs]
        res = {}

        for k, v in itertools.groupby(sorted(fwd + bck, key=lambda p:p[0]), lambda p: p[0]):
            x = list(v)
            res[k] = [y[1] for y in x]

        graph = nx.DiGraph(res)
        self.cycles = list(filter(lambda cycle: len(cycle) < 5, nx.simple_cycles(graph)))

    def weights(self):
        logging.info('Cycles', len(self.cycles))
        samples = lambda values: random.sample(values, min(30, len(values)))
        self.exchange.clear_cache()
        return [self._weight(cycle) for cycle in samples(self.cycles)]

    def _weight(self, cycle):
        mult = lambda values: reduce(lambda x, y: x * y, values)
        cycle_shifted = deque(cycle[:])
        cycle_shifted.rotate(-1)
        ops = [self.op(a, b) for (a, b) in zip(cycle, cycle_shifted)]
        ar = Arbitrage(ops, mult([op.rate if op.op == 'buy' else 1.0 / op.rate for op in ops]) / (0.9975 ** len(ops)))

        return ar

    def op(self, x, y, base=True):
        _op = 'buy' if base else 'sell'
        p = Pair(x, y) if base else Pair(y, x)
        (error, rate) = self.exchange.rate(p, _op)

        if error:
            return self.op(y, x, base=False)

        return Op(p, _op, rate)


def do_trade(exchange, trade):
    try:
        res = getattr(exchange, trade.op.op)(trade.op.pair, trade.qty, trade.op.rate)
        logging.info(res)
        return res
    except Exception as e:
        logging.info(e)
        return None


def _build_trades(ops, s0, acc=None):
    if not acc : acc = []
    if not ops: return acc
    op = ops[0]
    if op.op == 'buy':
        acc.append(Trade(op, (0.9975 * s0) / op.rate))
        return _build_trades(ops[1:], (0.9975 * s0) / op.rate, acc)
    acc.append(Trade(op, s0))
    return _build_trades(ops[1:], 0.9975 * s0 * op.rate, acc)

def _reorder(arbitrage):
    idx = 0
    for i, j in enumerate(arbitrage.ops):
        if j.pair.base in ['eth', 'btc', 'usdt'] and j.op == 'buy':
            idx = i
            break
    d = deque(arbitrage.ops)
    d.rotate(-1 * idx)
    return arbitrage._replace(ops=list(d)) 

def do_work(arbitro):
    try:
        logging.info('do_work')
        orders = arbitro.exchange.orders() 

        logging.info('orders', orders)

        if orders and not arbitro.exchange.debug:
            raise Exception("pending orders")

        arbitrages = arbitro.weights()
        arbitrage = min(arbitrages, key=lambda w: w.weight)
        arbitrage = _reorder(arbitrage)
        minimum = {'eth': 0.015,
                'btc': 0.0006,
                'usdt': 10 }

        trades = _build_trades(arbitrage.ops, minimum[arbitrage.ops[0].pair.base])

        if arbitrage.ops[0].pair.base in ['eth', 'btc', 'usdt'] and arbitrage.weight < 1.0:
            logging.info('Found: ', arbitrage)
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for trade in trades:
                    logging.info('submit')
                    f = executor.submit(do_trade, arbitro.exchange, trade)
                    futures.append(f)
                for f in futures:
                    r = f.result()
                    if not r['success']:
                        logging.info('ERROR')
        else:
            logging.info('not found', arbitrage)
    except Exception as e:
        logging.error(e)

    threading.Timer(120, do_work, (arbitro,)).start()



