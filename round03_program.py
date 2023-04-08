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
                 pt_max_delta=0.00025, 
                 pt_max_vol=4500000,
                 pt_pricenorm= { 'PINA_COLADAS':15000, 'COCONUTS':8000 },
                 mm_fair_price= { 'PEARLS':10000, 'BANANAS': 4780, 'BERRIES':4000, 'DIVING_GEAR':100000 },
                 mm_spread= { 'PEARLS':0, 'BANANAS':0, 'BERRIES':0, 'DIVING_GEAR':0 },
                 mm_avg_window= { 'PEARLS':20, 'BANANAS':7, 'BERRIES':4, 'DIVING_GEAR':5 },
                 dolphin_rally_time = 90000,
                 delta_obs_threshold = 5,
                 berries_strat = 'hybrid',
                 berries_rally_time = 100000,
                 ):
        
        self.pt_max_delta = pt_max_delta
        self.pt_max_vol = pt_max_vol
        self.pt_pricenorm = pt_pricenorm
        self.mm_fair_price = mm_fair_price
        self.mm_spread = mm_spread
        self.mm_avg_window = mm_avg_window
        self.mm_mid_prices = { prd:[ mm_fair_price[prd] ] * mm_avg_window[prd] for prd in mm_fair_price }
        self.dolphins = -1
        self.dolphin_timer = -1
        self.dolphin_obs_sgn = 1
        self.dolphin_rally_time = dolphin_rally_time
        self.delta_obs_threshold = delta_obs_threshold
        self.berries_strat = berries_strat
        self.berries_rally_time = berries_rally_time
        
        #Order limits, as set by the competition
        self.order_limits = { 'PEARLS':20, 
                              'BANANAS':20, 
                              'COCONUTS':600, 
                              'PINA_COLADAS':300, 
                              'BERRIES':250, 
                              'DIVING_GEAR':50, 
                            }
    
    def market_make(self, state, product):
        """
        Basic market making strategy.
        """
        order_depth: OrderDepth = state.order_depths[product]
        orders: list[Order] = []                    

        mid_price = calculate_mid_price( order_depth )                  
        if mid_price == None:
            mid_price = self.mm_mid_prices[product][-1]
        self.mm_mid_prices[product].pop(0)
        self.mm_mid_prices[product].append(mid_price)
        
        #Directional MM strat takes into account derivative also
        mean = np.mean(self.mm_mid_prices[product])
        deriv = (self.mm_mid_prices[product][-1] - self.mm_mid_prices[product][0])/self.mm_avg_window[product]
        
        self.mm_fair_price[product] = mean + deriv * self.mm_avg_window[product]/2
        
        
        acceptable_price = self.mm_fair_price[product]
        spread = self.mm_spread[product]

        for ask in order_depth.sell_orders:
            ask_volume = order_depth.sell_orders[ask]
            if ask < acceptable_price - spread:
#                 print("BUY", product, str(-ask_volume) + "x", ask)
                orders.append(Order(product, ask, abs(ask_volume)))                                            

        for bid in order_depth.buy_orders:
            bid_volume = order_depth.buy_orders[bid]
            if bid > acceptable_price + spread:
#                 print("SELL", product, str(bid_volume) + "x", bid)
                orders.append(Order(product, bid, -abs(bid_volume)))

        return orders
        
        
    def pair_trade(self, state, product1, product2):
        """
        Pair trading strategy.
        """
        mid_price_1 = calculate_mid_price( state.order_depths[product1] )
        mid_price_2 = calculate_mid_price( state.order_depths[product2] )        
        
        out_orders = { product1:[], product2:[] }
                
        if mid_price_1 != None and mid_price_2 != None:
            delta = mid_price_1 / self.pt_pricenorm[product1] - mid_price_2 / self.pt_pricenorm[product2]
            max_delta = self.pt_max_delta

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
            
        volume_to_trade = abs(quantity)
        for price in prices:
            volume = abs(orders[price])
            value = price * volume
            if volume < volume_to_trade:                            
                out_orders.append( Order(product, int(price), int( np.sign(quantity) * volume)) )                        
                volume_to_trade -= volume        
            else:
                out_orders.append( Order(product, int(price), int( np.sign(quantity) * volume_to_trade)) )                        
                
        return out_orders
        
        
    def get_value_best_price( self, state, product, quantity ):
        """
        Convenience function for matching incoming orders, starting with best prices offered.
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
        if state.position[product] != 0:            
            prod_pos = state.position[product]
            volume = -1 * np.sign(prod_pos) * min(abs(prod_pos),self.order_limits[product])            
            out_orders = self.place_orders_best_price( state, product, volume )
    
        return out_orders
    
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}

        print("position at open",state.position)        
        
        #Pure market making strategies
        for product in ['PEARLS', 'BANANAS']:
            result[product] = self.market_make( state, product )            
            
        brt = self.berries_rally_time
            
        if self.berries_strat == 'mm_only':
            result['BERRIES'] = self.market_make( state, 'BERRIES' )               
        elif self.berries_strat == 'hybrid':
            #Market make + time dependent trading                       
            if (state.timestamp < (500000 - brt)):  
                result['BERRIES'] = self.market_make( state, 'BERRIES' ) #MM at beginning and end of day
            elif (500000 - brt) < state.timestamp < 500000:
                result['BERRIES'] = self.place_orders_best_price( state, 'BERRIES', 250 ) # Buy during beginning of rally
            elif 500000 < state.timestamp < (500000 + brt):
                result['BERRIES'] = self.place_orders_best_price( state, 'BERRIES', -250 ) # Sell at the top                
#             elif state.timestamp > (500000 + brt):
#                 result['BERRIES'] = self.liquidate_position( state, 'BERRIES' ) # Sell at the top
            
            #Hold short position until end of day
    
    
        elif self.berries_strat == 'event_only':           
            if (500000 - brt) < state.timestamp < 500000:
                result['BERRIES'] = self.place_orders_best_price( state, 'BERRIES', 250 ) # Buy during beginning of rally
            elif 500000 < state.timestamp < (500000 + brt):               
                result['BERRIES'] = self.place_orders_best_price( state, 'BERRIES', -250 ) # Sell at the top
            elif state.timestamp > (500000 + brt):                               
                result['BERRIES'] = self.liquidate_position( state, 'BERRIES' )                
                                
                
        #Obsevation-based strategy
        if self.dolphins < 0:
            self.dolphins = state.observations['DOLPHIN_SIGHTINGS']
        else:
            delta_dolphins = state.observations['DOLPHIN_SIGHTINGS'] - self.dolphins
            self.dolphins = state.observations['DOLPHIN_SIGHTINGS']            
            
            if abs(delta_dolphins) >= self.delta_obs_threshold:
                print("dolphin observation",delta_dolphins)
                self.dolphin_timer = state.timestamp
                self.dolphin_obs_sgn = np.sign(delta_dolphins) #Set whether to sell or to buy
                            
            time_since_obs = state.timestamp - self.dolphin_timer if self.dolphin_timer > 0 else -1
            
            print(f"ts = {state.timestamp}, dolphin_delta = {delta_dolphins}, dolphin timer = {time_since_obs}, rally time = {self.dolphin_rally_time}")
            
            if 0 < time_since_obs < self.dolphin_rally_time : #Start buying at beginning of rally until order limit is reached
                result['DIVING_GEAR'] = self.place_orders_best_price( state, 'DIVING_GEAR', self.dolphin_obs_sgn * 50 )
#             elif self.dolphin_rally_time/2 < time_since_obs < self.dolphin_rally_time : #End of rally
#                 result['DIVING_GEAR'] = self.place_orders_best_price( state, 'DIVING_GEAR', -1 * self.dolphin_obs_sgn * 50 )
            elif time_since_obs > self.dolphin_rally_time : #After end of rally, liquidate position
                result['DIVING_GEAR'] = self.liquidate_position( state, 'DIVING_GEAR' )                
            
            
            if 'DIVING_GEAR' in result.keys(): 
                print(result['DIVING_GEAR'])
                    
        #Pair trading strategy
        if { 'PINA_COLADAS', 'COCONUTS' }.issubset( set(state.order_depths.keys()) ):
            pt_result = self.pair_trade( state, 'PINA_COLADAS', 'COCONUTS' )
            result.update( pt_result )
            
#         print("Trader result:",result)

        print("position at close",state.position)

        return result