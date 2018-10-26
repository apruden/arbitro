try:
    curr = ['btc', 'bcc', 'eth', 'ltc', 'usdt', 'dash', 'btg']
    Forcaster().init()
    arbitro = Arbitro(curr, Bittrex(curr, debug=False))
    do_work(arbitro)
except KeyboardInterrupt:
    pass
