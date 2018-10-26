import hmac
import logging
import networkx as nx

from bittrex_websocket.websocket_client import BittrexSocket
from collections import deque


class RatesBittrex(BittrexSocket):
    def __init__(self, pair):
        super(RatesBittrex, self).__init__()
        self.pair = pair
        self.bids = []
        self.asks = []
        self.buys = deque(maxlen=20)
        self.sells = deque(maxlen=20)
        self.subscribe()

    def on_orderbook_update(self, msg):
        bids = [x['Rate'] for x in msg['bids'] if x['Type'] == 0]
        asks = [x['Rate'] for x in msg['asks'] if x['Type'] == 0]
        for b in bids: self.bids.append(b) 
        for a in asks: self.asks.append(a) 

        bids = [x['Rate'] for x in msg['bids'] if x['Type'] == 1]
        asks = [x['Rate'] for x in msg['asks'] if x['Type'] == 1]

        for b in bids:
            try:
                self.bids.remove(b) 
            except ValueError as e:
                pass

        for a in asks:
            try:
                self.asks.remove(a) 
            except ValueError as e:
                pass
    
        self.bids = sorted(self.bids)[-20:]
        self.asks = sorted(self.asks)[:20]

    def on_trades(self, msg):
        buys = [x['Rate'] for x in msg['trades'] if x['OrderType'] == 'BUY']
        sells = [x['Rate'] for x in msg['trades'] if x['OrderType'] == 'SELL']
        for b in buys: self.buys.append(b)
        for s in sells: self.sells.append(s)

    def subscribe(self):
        #self.subscribe_to_trades([('%s-%s' % self.pair).upper()])
        self.subscribe_to_orderbook_update([('%s-%s' % self.pair).upper()])

    def get_rate(self):
        from statistics import stdev, mean

        while len(self.bids) < 2 and len(self.asks) < 2:
            logging.info('bids: ', len(self.bids), 'asks: ', len(self.asks))
            time.sleep(1)

        logging.info('READY')

        def validate(rates, other):
            #return len(rates) > 1 and stdev(rates)/mean(rates) < 0.05
            return True

        if validate(self.sells, self.asks) and validate(self.buys, self.bids):
            ask = min(self.asks)
            bid = max(self.bids)
            return {'Ask': ask,'Bid': bid}


class Exchange:
    def __init__(self):
        pass

    def _get(self, endpoint):
        return requests.get(self.base + endpoint).json()

    def buy(self, pair, qty, rate):
        logging.info('buy', pair, qty, rate)
    
    def sell(self, pair, qty, rate):
        logging.info('sell', pair, qty, rate)


class Bittrex(Exchange):
    def __init__(self, curr, debug=True):
        super(Bittrex, self).__init__()
        self._cache = {}
        self.base = 'https://bittrex.com/api/v1.1'
        self.key = ''
        self.secret = ''
        self.debug = debug
        self._markets = self.markets()
        self._curr = curr
        self._ws = self.init_ws()

    def init_ws(self, pairs):
        pass

    def markets(self):
        if self._markets: return self._markets
        res = self._get('/public/getmarkets')
        self._markets = [Pair(quote=m['MarketCurrency'].lower(), base=m['BaseCurrency'].lower()) for m in res['result']]
        return self._markets

    def clear_cache(self):
        logging.info('cache cleared')
        self._cache = {}

    def ticker(self, pair):
        res = self._get('/public/getticker?market=%s-%s' % (pair.base, pair.quote))
        return res


    def _forecast_rate(self, pair):
        #data = self.order_book('%s-%s' % pair)
        #f = Forecaster()
        #f.init(data)
        #res = f.predict(10)

        #res = self.ticker(pair)

        if pair not in self._ws:
            self._ws[pair] = RatesBittrex(pair)

        return self._ws[pair].get_rate()

        #return res['result']#dict(Ask=res[0], Bid=res[1])

    def rate(self, pair, op):
        if pair not in self._markets: return (True, None)

        if pair in self._cache:
            res = self._cache[pair]
        else:
            res = self._forecast_rate(pair)
            self._cache[pair] = res

        logging.info('pair: ', pair, 'res: ', res)

        if not res: return (False, res)

        k = 'Ask' if op == 'buy' else 'Bid'
        factor = 0.998 if op == 'buy' else 1.002
        rate = res[k] * factor

        return (False, rate)

    def buy(self, pair, qty, rate):
        logging.info('buy', pair, qty, rate)
        if self.debug: return
        m = '%s-%s' % pair
        res = self._get_signed('/market/buylimit', market=m.upper(), quantity=qty, rate=rate)
        return res

    def orders(self):
        res = self._get_signed('/market/getopenorders')
        return res['result']

    def market_history(self, market):
        res = self._get('/public/getmarkethistory', market=market.upper())
        rev = lambda l: list(reversed(l))
        sell = rev([o['Price'] for o in res['result'] if o['OrderType'] == 'Sell'])
        buy = rev([o['Price'] for o in res['result'] if o['OrderType'] == 'Buy'])

        return {'Sell': sell, 'Buy': buy}

    def orderbook(self, market):
        res = self._get('/public/getorderbook', market=market.upper())
        rev = lambda l: list(reversed(l))
        sell = rev([o['Rate'] for o in res['result']['sell']])
        buy = rev([o['Rate'] for o in res['result']['buy']])

        return {'Sell': sell, 'Buy': buy}

    def _get_signed(self, endpoint, **kwargs):
        query = kwargs
        query.update(dict(apikey=self.key, nonce=int(time.time())))
        uri = '%s%s?%s' % (self.base, endpoint, urlencode(query))
        res = requests.get(uri, headers=dict(apisign=self._sign(uri)))
        return res.json()

    def _sign(self, uri):
        h = hmac.new(self.secret.encode('UTF-8'), uri.encode('UTF-8'), 'SHA512')
        return h.hexdigest()

    def sell(self, pair, qty, rate):
        logging.info('sell', pair, qty, rate)
        if self.debug: return
        m = '%s-%s' % pair
        res = self._get_signed('/market/selllimit', market=m.upper(), quantity=qty, rate=rate)
        return res

    def balance(self):
        return self._get_signed('/account/getbalances')


class Yobit(Exchange):
    def __init__(self):
        super(Yobit, self).__init__()
        self._depth_cache = {}
        self.base = 'https://yobit.net/api/3'

    def markets(self):
        info = self._get('/info')
        return map(lambda p: Pair(quote=p[1], base=p[0]), [p.split('_') for p in info['pairs'].keys()])

    def rate(self, pair, base):
        raise Exception("base")
        key = '%s_%s' % pair
        if pair in self._depth_cache:
            return self._depth_cache[pair]
        res = self._get('/depth/%s' % (key,))

        if 'error' in res:
            (error, rate) = (True, None)
        else:
            (error, rate) = (False, res[key]['bids'][0][0])
        self._depth_cache[pair] = (error, rate)
        return (error, rate)

