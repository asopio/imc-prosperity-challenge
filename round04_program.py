import numpy as np
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order


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
                 pt_max_deltas= { 'PINA_COLADAS-COCONUTS': 0.00025, 
                                  'picnic1-picnic2': 0.0017,
                                  'pina_coladas-coconuts': 0.00025 }, 
                 pt_min_deltas= { 'PINA_COLADAS-COCONUTS': 0.0001, 
                                  'picnic1-picnic2': 0.0001,
                                  'pina_coladas-coconuts': 0.0001},                 
                 pt_max_vol=4500000,
                 pt_pricenorm= { 'PINA_COLADAS':15000, 'COCONUTS':8000, 
                                 'picnic1':74025, 'picnic2':73650,
                                 'pina_coladas':15000, 'coconuts':8000,
                                 },                                  
                 pt_avg_window= { 'PINA_COLADAS-COCONUTS': 1, 
                                  'picnic1-picnic2': 1,
                                  'pina_coladas-coconuts': 1},
                 pt_liquidate=False,
                 mm_fair_price= { 'PEARLS':10000, 'BANANAS': 4780, 'BERRIES':4000 },
                 mm_spread= { 'PEARLS':0, 'BANANAS':0, 'BERRIES':0 },
                 mm_avg_window= { 'PEARLS':20, 'BANANAS':7, 'BERRIES':4 },
                 mm_fp_adjustment= { 'PEARLS':0, 'BANANAS':0, 'BERRIES':-0.6 },                 
                 dolphin_rally_time = 90000,
                 delta_obs_threshold = 5,
                 berries_strat = 'hybrid',
                 berries_rally_time = 100000,
                 pcpt_oldimp = False,
                 order_limits = { 'PEARLS':20, 
                                  'BANANAS':20, 
                                  'COCONUTS': 563, #adjust limit to pair integer multiple       
                                  'PINA_COLADAS':300, 
                                  'BERRIES':250, 
                                  'DIVING_GEAR':50,
                                  'BAGUETTE': 140,                             
                                  'DIP': 280,
                                  'UKULELE': 70,                             
                                  'PICNIC_BASKET': 70,
                                }
                 ):
        
        self.pt_max_deltas = pt_max_deltas
        self.pt_min_deltas = pt_min_deltas        
        self.pt_max_vol = pt_max_vol
        self.pt_pricenorm = pt_pricenorm
        self.pt_avg_window = pt_avg_window
        self.pt_deltas = { pair:[ 0 ] * pt_avg_window[pair] for pair in pt_avg_window }
        self.pt_liquidate = pt_liquidate
        self.pcpt_oldimp = pcpt_oldimp        
        
        self.mm_fair_price = mm_fair_price
        self.mm_spread = mm_spread
        self.mm_avg_window = mm_avg_window
        self.mm_fp_adjustment = mm_fp_adjustment
        self.mm_mid_prices = { prd:[ mm_fair_price[prd] ] * mm_avg_window[prd] for prd in mm_fair_price }
        self.dolphins = -1
        self.dolphin_timer = -1
        self.dolphin_obs_sgn = 1
        self.dolphin_rally_time = dolphin_rally_time
        self.delta_obs_threshold = delta_obs_threshold
        self.berries_strat = berries_strat
        self.berries_rally_time = berries_rally_time
        
        self.product_baskets = { "picnic1":{ "PICNIC_BASKET":1 }, 
                                 "picnic2":{ "DIP":4, "BAGUETTE":2, "UKULELE":1 },
                                 "pina_coladas":{ "PINA_COLADAS":1 },
                                 "coconuts":{ "COCONUTS":1 } }            
        
        #Order limits, as set by the competition
        self.order_limits = order_limits
        
    def market_make(self, state, product):
        """
        Basic market making strategy.
        """
        order_depth: OrderDepth = state.order_depths[product]
        orders: list[Order] = []                    

        mid_price = calculate_mid_price( order_depth )                  
        if mid_price == None:
            mid_price = self.mm_mid_prices[product][-1]
            
        if state.timestamp == 0:
            self.mm_mid_prices[product] = [ mid_price ] * self.mm_avg_window[product]
            
        self.mm_mid_prices[product].pop(0)
        self.mm_mid_prices[product].append(mid_price)
        
        #Directional MM strat takes into account derivative also
        mean = np.mean(self.mm_mid_prices[product])
#         deriv = (self.mm_mid_prices[product][-1] - self.mm_mid_prices[product][0])/self.mm_avg_window[product]        
#         self.mm_fair_price[product] = mean + deriv * self.mm_avg_window[product]/2
        
        self.mm_fair_price[product] = mean + self.mm_fp_adjustment[product]
        
        acceptable_price = self.mm_fair_price[product]
        spread = self.mm_spread[product]

        sos = order_depth.sell_orders
        bos = order_depth.buy_orders
        
        tot_ask_vol = int( np.sum([ sos[ask] for ask in sos if ask < (acceptable_price - spread) ]) )
        tot_bid_vol = int( np.sum([ bos[bid] for bid in bos if bid > (acceptable_price + spread) ]) )
                
        my_buy_orders = self.place_orders_best_price( state, product, abs(tot_ask_vol) )
        my_sell_orders = self.place_orders_best_price( state, product, -abs(tot_bid_vol) )
                        
        orders += my_buy_orders
        orders += my_sell_orders
        
#         for ask in order_depth.sell_orders:
#             ask_volume = order_depth.sell_orders[ask]
#             if ask < acceptable_price - spread:
# #                 print("BUY", product, str(-ask_volume) + "x", ask)
#                 orders.append(Order(product, ask, abs(ask_volume)))                                            

#         for bid in order_depth.buy_orders:
#             bid_volume = order_depth.buy_orders[bid]
#             if bid > acceptable_price + spread:
# #                 print("SELL", product, str(bid_volume) + "x", bid)
#                 orders.append(Order(product, bid, -abs(bid_volume)))

        return orders
    
        
    def calculate_mid_price_basket(self, state, basket):                
        odp = state.order_depths
        pbs = self.product_baskets
        mid_prices = [ calculate_mid_price(odp[prd]) * pbs[basket][prd] for prd in pbs[basket] ]
        mid_price_basket = np.sum(mid_prices) 
        
        return mid_price_basket 
        
    def get_tot_bidask_basket(self, state, basket, bidask="bid"):
    
        odp = state.order_depths
        bask_dic = self.product_baskets[basket]
        
        if bidask == "bid":
            num_baskets_products = [ np.sum(list(odp[prd].buy_orders.values())) // bask_dic[prd] for prd in bask_dic ]
        elif bidask == "ask":
            num_baskets_products = [ np.sum(list(odp[prd].sell_orders.values())) // bask_dic[prd] for prd in bask_dic ]
        
        num_baskets = min(np.abs(num_baskets_products))
        
        sgn = {"bid":1, "ask":-1}[bidask]
        gvbp = self.get_value_best_price
        
        value_baskets = [ gvbp( state, prd, sgn * num_baskets * bask_dic[prd] ) for prd in bask_dic ]
        tot_value = np.sum(value_baskets)
        
        avg_price = abs(tot_value) / num_baskets
        
        return tot_value, avg_price
        
    def place_basket_orders_best_price( self, state, basket, quantity ):
        
        bask_dic = self.product_baskets[basket]        
        out_orders = {}        
        for product in bask_dic:
            out_orders[product] = self.place_orders_best_price( state, product, quantity * bask_dic[product] )
        
        return out_orders
        
    def pair_trade_baskets(self, state, basket1, basket2):
        """
        Pair trading strategy for two baskets of products.        
        """        
        out_orders = {}
        
        mid_price_1 = self.calculate_mid_price_basket( state, basket1 )
        mid_price_2 = self.calculate_mid_price_basket( state, basket2 )        

#         print(f"Basket mid prices: {basket1}={mid_price_1}, {basket2}={mid_price_2}")
        
        if mid_price_1 != None and mid_price_2 != None:                       
            delta = mid_price_1 / self.pt_pricenorm[basket1] - mid_price_2 / self.pt_pricenorm[basket2]
            
            self.pt_deltas["{}-{}".format(basket1,basket2)].pop(0)
            self.pt_deltas["{}-{}".format(basket1,basket2)].append(delta)
            avg_delta = np.mean(self.pt_deltas["{}-{}".format(basket1,basket2)])
            
            max_delta = self.pt_max_deltas["{}-{}".format(basket1,basket2)]
            min_delta = self.pt_min_deltas["{}-{}".format(basket1,basket2)]
            
            short_basket = basket1 if delta > 0 else basket2
            long_basket = basket2 if delta > 0 else basket1
            
#             print(f"Basket delta={delta}(max {max_delta})")
            
            if abs(avg_delta) > max_delta:
                tot_ask_value, avg_ask_price = self.get_tot_bidask_basket( state, long_basket, 'ask' )
                tot_bid_value, avg_bid_price = self.get_tot_bidask_basket( state, short_basket, 'bid' )

#                 print(f"Basket values = ({tot_ask_value} ask, {tot_bid_value} bid)")
                
                trade_value = min( abs(tot_ask_value), abs(tot_bid_value) )
                
                baskets_to_buy = int( trade_value / avg_ask_price )
                baskets_to_sell = int( trade_value / avg_bid_price )
            
#                 print(f"Buying(selling) {baskets_to_buy}({baskets_to_sell}) baskets")
            
                long_orders = self.place_basket_orders_best_price( state, long_basket, baskets_to_buy )
                short_orders = self.place_basket_orders_best_price( state, short_basket, -baskets_to_sell )
            
#                 print("buy orders",long_orders)
#                 print("sell orders",short_orders)                
            
                out_orders.update( long_orders )
                out_orders.update( short_orders )
           
            elif self.pt_liquidate and abs(avg_delta) < min_delta:
                for prod1 in self.product_baskets[basket1]:
                    out_orders[prod1] = self.liquidate_position( state, prod1 )
                for prod2 in self.product_baskets[basket2]:
                    out_orders[prod2] = self.liquidate_position( state, prod2 )
                
        return out_orders
        
        
    def pair_trade(self, state, product1, product2):
        """
        Pair trading strategy.
        """
        mid_price_1 = calculate_mid_price( state.order_depths[product1] )
        mid_price_2 = calculate_mid_price( state.order_depths[product2] )        
        
        out_orders = { product1:[], product2:[] }
                
        if mid_price_1 != None and mid_price_2 != None:
            delta = mid_price_1 / self.pt_pricenorm[product1] - mid_price_2 / self.pt_pricenorm[product2]
            max_delta = self.pt_max_deltas["{}-{}".format(product1,product2)]

            short_product = product1 if delta > 0 else product2
            long_product = product2 if delta > 0 else product1

            if abs(delta) > max_delta:
                sell_orders = state.order_depths[long_product].sell_orders
                buy_orders = state.order_depths[short_product].buy_orders

                tot_ask_value = -1.0 * np.sum([ sell_orders[ask] * ask for ask in sell_orders ])
                tot_bid_value = np.sum([ buy_orders[bid] * bid for bid in buy_orders ])

                trade_value = min( tot_ask_value, tot_bid_value, self.pt_max_vol )

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
#                         my_buy_orders.append( Order(long_product, int(ask), int(ask_volume)) )
                        out_orders[long_product].append( Order(long_product, int(ask), int(ask_volume)) )                        
                        value_to_buy -= ask_value

                    elif value_to_buy > 0:
                        units_to_buy = int( value_to_buy / ask )
                        # print("BUY",long_product,ask,'x',units_to_buy)       
#                         my_buy_orders.append( Order(long_product, int(ask), int(units_to_buy)) )
                        out_orders[long_product].append( Order(long_product, int(ask), int(units_to_buy)) )                        
                        value_to_buy = 0

                for bid in bid_prices:
                    bid_volume = abs(buy_orders[bid])
                    bid_value = bid * bid_volume 

                    if bid_value < value_to_sell:
                        # print("SELL",short_product,bid,'x',bid_volume)
#                         my_sell_orders.append( Order(short_product, int(bid), -int(bid_volume)))
                        out_orders[short_product].append( Order(short_product, int(bid), -int(bid_volume)))                        
                        value_to_sell -= bid_value
        
                    elif value_to_sell > 0:
                        units_to_sell = int( value_to_sell / bid )
                        # print("SELL",short_product,bid,'x',units_to_sell)
#                         my_sell_orders.append( Order(short_product, int(bid), -int(units_to_sell)) )
                        out_orders[short_product].append( Order(short_product, int(bid), -int(units_to_sell)) )                        
                        value_to_sell = 0
                        
            
#             elif self.pt_liquidate:                
#                 out_orders[product1] = self.liquidate_position( state, product1 )
#                 out_orders[product2] = self.liquidate_position( state, product2 )
                
                
        return out_orders

    def place_orders_best_price( self, state, product, quantity ):
        """
        Convenience function for matching incoming orders, starting with best prices offered.
        """
        out_orders = []
        
        ods = state.order_depths[product]
        orders = ods.sell_orders if quantity > 0 else ods.buy_orders
                
        prices = np.array(list(orders.keys())) 
        prices = np.sort(prices) #sort low to high if buying
        if quantity < 0: 
            prices = np.sort(prices)[::-1] #high to low if selling

        #Make sure orders stay within order limits
        volume_to_trade = abs(quantity)        
        if product in state.position:
            pos = state.position[product]
            ol = self.order_limits[product]
            if abs(pos + quantity) > ol:                        
                volume_to_trade = self.order_limits[product] - abs(pos) 

        for price in prices:
            volume = abs(orders[price])
            value = price * volume
            if volume <= volume_to_trade:                            
                out_orders.append( Order(product, int(price), int( np.sign(quantity) * volume)) )                        
                volume_to_trade -= volume
            elif volume_to_trade > 0:
                out_orders.append( Order(product, int(price), int( np.sign(quantity) * volume_to_trade)) )                        
                volume_to_trade = 0
                
        return out_orders
        
    def get_value_best_price( self, state, product, quantity ):
        """
        Calclulates values of quantity at best prices offered (equivalent to place_orders_best_price without placing orders).
        """        
        total_value = 0
        
        ods = state.order_depths[product]
        orders = ods.sell_orders if quantity > 0 else ods.buy_orders
                
        prices = np.array(list(orders.keys())) 
        prices = np.sort(prices) #sort low to high if buying
        if quantity < 0: 
            prices = np.sort(prices)[::-1] #high to low if selling
            
        volume_to_trade = abs(quantity)
        for price in prices:
            volume = abs(orders[price])
            value = price * volume
            if volume < volume_to_trade:                            
#                 out_orders.append( Order(product, int(price), int( np.sign(quantity) * volume)) )     
                total_value += price * np.sign(quantity) * volume
                volume_to_trade -= volume        
            else:
#                 out_orders.append( Order(product, int(price), int( np.sign(quantity) * volume_to_trade)) )                        
                total_value += price * np.sign(quantity) * volume_to_trade
                
        return total_value    
    
    def liquidate_position( self, state, product ):
        """
        Keeps selling/buying of product at best prices until position is 0. 
        """        
        out_orders = []        
        if product not in state.position.keys():
            return out_orders
        
        if state.position[product] != 0:            
            prod_pos = state.position[product]
            volume = -1 * np.sign(prod_pos) * abs(prod_pos)            
            out_orders = self.place_orders_best_price( state, product, volume )
    
        return out_orders
    
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}

#         print("position at open",state.position)        
        
        print( "listings", state.listings )
        print( "own_trades", state.own_trades )
        print( "market_trades", state.market_trades )
        print( "sell orders", { p:state.order_depths[p].sell_orders for p in state.order_depths } )
        print( "buy orders", { p:state.order_depths[p].buy_orders for p in state.order_depths } )        
    
        #Pure market making strategies
        for product in ['PEARLS', 'BANANAS']:
            result[product] = self.market_make( state, product )            
            
        brt = self.berries_rally_time
            
        if self.berries_strat == 'mm_only':
            result['BERRIES'] = self.market_make( state, 'BERRIES' )          
            
            
        elif self.berries_strat == 'hybrid':
            #Market make + time dependent trading                       
            if (state.timestamp <= (500000 - brt)) or (state.timestamp > (500000 + brt)) :  
                result['BERRIES'] = self.market_make( state, 'BERRIES' ) #MM at beginning and end of day
            elif (500000 - brt) <= state.timestamp < 500000:
                result['BERRIES'] = self.place_orders_best_price( state, 'BERRIES', 250 ) # Buy during beginning of rally
            elif 500000 <= state.timestamp < (500000 + brt - 2000):
                result['BERRIES'] = self.place_orders_best_price( state, 'BERRIES', -250 ) # Sell at the top                
            elif state.timestamp >= (500000 + brt - 2000):
                result['BERRIES'] = self.liquidate_position( state, 'BERRIES' ) # Liquidate position before MM resumes
            
            
        elif self.berries_strat == 'event_only':           
            if (500000 - brt) <= state.timestamp < 500000:
                result['BERRIES'] = self.place_orders_best_price( state, 'BERRIES', 250 ) # Buy during beginning of rally
            elif 500000 <= state.timestamp < (500000 + brt):               
                result['BERRIES'] = self.place_orders_best_price( state, 'BERRIES', -250 ) # Sell at the top
            elif state.timestamp >= (500000 + brt):                               
                result['BERRIES'] = self.liquidate_position( state, 'BERRIES' )                
                                
                
        #Obsevation-based strategy
        if self.dolphins < 0:
            self.dolphins = state.observations['DOLPHIN_SIGHTINGS']
        else:
            delta_dolphins = state.observations['DOLPHIN_SIGHTINGS'] - self.dolphins
            self.dolphins = state.observations['DOLPHIN_SIGHTINGS']            
            
            if abs(delta_dolphins) >= self.delta_obs_threshold:
                self.dolphin_timer = state.timestamp
                self.dolphin_obs_sgn = np.sign(delta_dolphins) #Set whether to sell or to buy
                            
            time_since_obs = state.timestamp - self.dolphin_timer if self.dolphin_timer > 0 else -1
            
#             print(f"ts = {state.timestamp}, dolphin_delta = {delta_dolphins}, dolphin timer = {time_since_obs}, rally time = {self.dolphin_rally_time}")
            
            if 0 < time_since_obs < self.dolphin_rally_time : #Start buying at beginning of rally until order limit is reached
                result['DIVING_GEAR'] = self.place_orders_best_price( state, 'DIVING_GEAR', self.dolphin_obs_sgn * 50 )
#             elif self.dolphin_rally_time/2 < time_since_obs < self.dolphin_rally_time : #End of rally
#                 result['DIVING_GEAR'] = self.place_orders_best_price( state, 'DIVING_GEAR', -1 * self.dolphin_obs_sgn * 50 )
            elif time_since_obs > self.dolphin_rally_time : #After end of rally, liquidate position
                result['DIVING_GEAR'] = self.liquidate_position( state, 'DIVING_GEAR' )                
            
            
                    
        # Pair trading strategy
        if { 'PINA_COLADAS', 'COCONUTS' }.issubset( set(state.order_depths.keys()) ):            
            if self.pcpt_oldimp:
                pt_result = self.pair_trade( state, 'PINA_COLADAS', 'COCONUTS' )
                result.update( pt_result )
            else:
                pt_result = self.pair_trade_baskets( state, 'pina_coladas', 'coconuts' )
                result.update( pt_result )
            
        # Pair trading using product baskets
        basket_pt_products = { "PICNIC_BASKET", "DIP", "BAGUETTE", "UKULELE" }
        if basket_pt_products.issubset( set(state.order_depths.keys()) ):
            pt_basket_result = self.pair_trade_baskets( state, 'picnic1', 'picnic2' )
            result.update( pt_basket_result )
                    
#         print("Trader output", result)
          
        return result