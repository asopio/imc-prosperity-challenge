import numpy as np
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order


previous_state_mid_price = {}

my_pnl =  { 'cash':0, 'PEARLS':0, 'BANANAS':0 }
mid_prices = { 'cash':1.0, 'BANANAS':0, 'PEARLS':0}


def calculate_mid_price( order_depth: OrderDepth ):

    wsum_asks = np.sum([ k * abs(order_depth.sell_orders[k]) for k in order_depth.sell_orders ])
    tot_askvol = np.sum([ abs(order_depth.sell_orders[k]) for k in order_depth.sell_orders ])

    wsum_bids = np.sum([ k * abs(order_depth.buy_orders[k]) for k in order_depth.buy_orders ])
    tot_bidvol = np.sum([ abs(order_depth.buy_orders[k]) for k in order_depth.buy_orders ])

    avg_ask = wsum_asks / tot_askvol
    avg_bid = wsum_bids / tot_bidvol

    mid_price = (wsum_bids + wsum_asks) / (tot_bidvol + tot_askvol)

    if (tot_bidvol + tot_askvol) > 0:
        return mid_price
    else:
        return None


def place_sell_order( product, price, quantity, verbose=True ):
    if verbose:
        print("SELL", product, str(quantity) + "x", price)
    return Order(product, price, -abs(quantity))


def place_buy_order( product, price, quantity, verbose=True ):
    if verbose:
        print("BUY", product, str(quantity) + "x", price)
    return Order(product, price, abs(quantity))


class Trader:
    def __init__(self, 
                 max_delta=0.00025, 
                 max_pairtrade_vol=2400000,
                 pearl_fair_price=10000,
                 pearl_spread=0,
                 banana_fair_price=4780,
                 banana_spread=0 ):
        self.max_delta = max_delta
        self.max_pairtrade_vol = max_pairtrade_vol
        self.pearl_fair_price = pearl_fair_price
        self.pearl_spread = pearl_spread        
        self.banana_fair_price = banana_fair_price
        self.banana_spread = banana_spread
        
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}

        #Market making for pearls and bananas
        
        # Iterate over all the keys (the available products) contained in the order dephts
        for product in ['PEARLS','BANANAS']:

            # Check if the current product is the 'PEARLS' product, only then run the order logic
            # if product in [ 'BANANAS' ]:
            # if product in [ 'PEARLS' ]:

                # Retrieve the Order Depth containing all the market BUY and SELL orders for PEARLS
                order_depth: OrderDepth = state.order_depths[product]

                # Initialize the list of Orders to be sent as an empty list
                orders: list[Order] = []
                    
                mid_price = calculate_mid_price( order_depth )

                #Only update price if there are bids and asks!
                if mid_price != None:
                    mid_prices[product] = mid_price

                # Define a fair value for the PEARLS.
                # Note that this value of 1 is just a dummy value, you should likely change it!                
                # it = state.timestamp // 100 - 1

                acceptable_price = { 'PEARLS':self.pearl_fair_price, 'BANANAS':self.banana_fair_price }[product]
                # acceptable_price = { 'PEARLS':10000, 'BANANAS':4890 }[product]

                spread = { 'PEARLS':self.pearl_spread, 'BANANAS':self.banana_spread }[product]

                for ask in order_depth.sell_orders:
                    ask_volume = order_depth.sell_orders[ask]
                    if ask < acceptable_price - spread:
                        print("BUY", product, str(-ask_volume) + "x", ask)
                        orders.append(Order(product, ask, abs(ask_volume)))                                            
                        my_pnl['cash'] -= ask * abs(ask_volume)
                        my_pnl[product] += abs(ask_volume)

                for bid in order_depth.buy_orders:
                    bid_volume = order_depth.buy_orders[bid]
                    if bid > acceptable_price + spread:
                        print("SELL", product, str(bid_volume) + "x", bid)
                        orders.append(Order(product, bid, -abs(bid_volume)))
                        my_pnl['cash'] += bid * abs(bid_volume)
                        my_pnl[product] -= abs(bid_volume)

                result[product] = orders

        #Pair trading strategy
        if { 'PINA_COLADAS', 'COCONUTS' }.issubset( set(state.order_depths.keys()) ):

            mid_price_p = calculate_mid_price( state.order_depths['PINA_COLADAS'] )
            mid_price_c = calculate_mid_price( state.order_depths['COCONUTS'] )

#             print(f"Mid price: PINA_COLADAS {mid_price_p} COCONUTS {mid_price_c}")

            if mid_price_p != None and mid_price_c != None:
                delta = mid_price_p / 15000 - mid_price_c / 8000
                max_delta = self.max_delta

#                 print("PINA_COLADAS - COCONUTS delta =",delta)

                short_product = 'PINA_COLADAS' if delta > 0 else 'COCONUTS'
                long_product = 'COCONUTS' if delta > 0 else 'PINA_COLADAS'

                if abs(delta) > max_delta:
                    my_buy_orders: list[Order] = []
                    my_sell_orders: list[Order] = []

                    sell_orders = state.order_depths[long_product].sell_orders
                    buy_orders = state.order_depths[short_product].buy_orders

                    tot_ask_value = -1.0 * np.sum([ sell_orders[ask] * ask for ask in sell_orders ])
                    tot_bid_value = np.sum([ buy_orders[bid] * bid for bid in buy_orders ])

                    trade_value = min( tot_ask_value, tot_bid_value, self.max_pairtrade_vol )

#                     print("tot_ask_value:", tot_ask_value)
#                     print("tot_bid_value:", tot_bid_value)
#                     print("trade_value:", trade_value)

#                     print("sell_orders",sell_orders)
#                     print("buy_orders",buy_orders)

                    value_to_buy = trade_value
                    value_to_sell = trade_value

                    ask_prices = np.array(list(sell_orders.keys())) #lowest first
                    bid_prices = np.array(list(buy_orders.keys())) #highest first
                    ask_prices = np.sort(ask_prices)
                    bid_prices = np.sort(bid_prices)[::-1]

                    for ask in ask_prices:
                        ask_volume = abs(sell_orders[ask])
                        ask_value = ask * ask_volume

                        if ask_value < value_to_buy:
                            # print("BUY",long_product,ask,'x',ask_volume)
                            my_buy_orders.append( Order(long_product, int(ask), int(ask_volume)) )
                            value_to_buy -= ask_value

                        elif value_to_buy > 0:
                            units_to_buy = int( value_to_buy / ask )
                            # print("BUY",long_product,ask,'x',units_to_buy)       
                            my_buy_orders.append( Order(long_product, int(ask), int(units_to_buy)) )
                            value_to_buy = 0

                    for bid in bid_prices:
                        bid_volume = abs(buy_orders[bid])
                        bid_value = bid * bid_volume 

                        if bid_value < value_to_sell:
                            # print("SELL",short_product,bid,'x',bid_volume)
                            my_sell_orders.append( Order(short_product, int(bid), -int(bid_volume)))
                            value_to_sell -= bid_value
                        elif value_to_sell > 0:
                            units_to_sell = int( value_to_sell / bid )
                            # print("SELL",short_product,bid,'x',units_to_sell)
                            my_sell_orders.append( Order(short_product, int(bid), -int(units_to_sell)) )
                            value_to_sell = 0

                    result[long_product] = my_buy_orders
                    result[short_product] = my_sell_orders


        print("Trader result:",result)
#         my_pnl_value = np.sum([ my_pnl[key] * mid_prices[key] for key in my_pnl])
#         print("PNL at end of turn:", my_pnl, "value = ", my_pnl_value)

        return result