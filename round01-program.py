import numpy as np
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order


previous_state_mid_price = {}

my_pnl =  { 'cash':0, 'PEARLS':0, 'BANANAS':0 }
mid_prices = { 'cash':1.0, 'BANANAS':0, 'PEARLS':0}


class Trader:
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}

        print("- - - -")
        for product in state.order_depths:
            print("Orders for product:",product)
            print("Buy:",state.order_depths[product].buy_orders)
            print("Sell:",state.order_depths[product].sell_orders)
        print("Market trades:",state.market_trades)
        print("Own trades:",state.own_trades)
        print("Position:",state.position)
        print("- - - -")

        # Iterate over all the keys (the available products) contained in the order dephts
        for product in state.order_depths.keys():

            # Check if the current product is the 'PEARLS' product, only then run the order logic
            # if product in [ 'BANANAS' ]:
            # if product in [ 'PEARLS' ]:

                # Retrieve the Order Depth containing all the market BUY and SELL orders for PEARLS
                order_depth: OrderDepth = state.order_depths[product]

                # Initialize the list of Orders to be sent as an empty list
                orders: list[Order] = []

                wsum_asks = np.sum([ k * order_depth.sell_orders[k] for k in order_depth.sell_orders ])
                tot_askvol = np.sum([ order_depth.sell_orders[k] for k in order_depth.sell_orders ])

                wsum_bids = np.sum([ k * order_depth.buy_orders[k] for k in order_depth.buy_orders ])
                tot_bidvol = np.sum([ order_depth.buy_orders[k] for k in order_depth.buy_orders ])

                avg_ask = wsum_asks / tot_askvol
                avg_bid = wsum_bids / tot_bidvol

                mid_price = (wsum_bids + wsum_asks) / (tot_bidvol + tot_askvol)

                if (tot_bidvol + tot_askvol) != 0:
                    #Only update price if there are bids and asks!
                    mid_prices[product] = mid_price

                # Define a fair value for the PEARLS.
                # Note that this value of 1 is just a dummy value, you should likely change it!                
                # it = state.timestamp // 100 - 1

                acceptable_price = { 'PEARLS':10000, 'BANANAS':4862 }[product]
                # acceptable_price = { 'PEARLS':10000, 'BANANAS':4890 }[product]

                spread = { 'PEARLS':0, 'BANANAS':3 }[product]

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


                # # # If statement checks if there are any SELL orders in the PEARLS market
                # if len(order_depth.sell_orders) > 0:
                #     # Sort all the available sell orders by their price,
                #     # and select only the sell order with the lowest price
                #     best_ask = min(order_depth.sell_orders.keys())
                #     best_ask_volume = order_depth.sell_orders[best_ask]

                #     # Check if the lowest ask (sell order) is lower than the above defined fair value
                #     if best_ask <= acceptable_price - spread:

                #         # In case the lowest ask is lower than our fair value,
                #         # This presents an opportunity for us to buy cheaply
                #         # The code below therefore sends a BUY order at the price level of the ask,
                #         # with the same quantity
                #         # We expect this order to trade with the sell order
                #         print("BUY", product, str(-best_ask_volume) + "x", best_ask)
                #         orders.append(Order(product, best_ask, -best_ask_volume))

                # # The below code block is similar to the one above,
                # # the difference is that it find the highest bid (buy order)
                # # If the price of the order is higher than the fair value
                # # This is an opportunity to sell at a premium
                # if len(order_depth.buy_orders) != 0:
                #     best_bid = max(order_depth.buy_orders.keys())
                #     best_bid_volume = order_depth.buy_orders[best_bid]
                #     if best_bid >= acceptable_price + spread:
                #         print("SELL", product, str(best_bid_volume) + "x", best_bid)
                #         orders.append(Order(product, best_bid, -best_bid_volume))

                # Add all the above the orders to the result dict
                result[product] = orders

                # Return the dict of orders
                # These possibly contain buy or sell orders for PEARLS
                # Depending on the logic above


            # if product == "BANANAS":

            #     order_depth: OrderDepth = state.order_depths[product]
            #     orders: list[Order] = []

          

            #     if product in previous_state_mid_price:
            #         price_chg_mid = mid_price - previous_state_mid_price[product]

            #         acceptable_price = mid_price

            #         print("mid_price = ", mid_price)

            #         best_ask = min(order_depth.sell_orders.keys())
            #         best_ask_volume = order_depth.sell_orders[best_ask]
            #         price_chg_ask = best_ask - previous_state_mid_price[product]

            #         print("best_ask =", best_ask)

            #         if best_ask < acceptable_price:
            #             print("BUY", product, str(-best_ask_volume) + "x", best_ask)
            #             orders.append(Order(product, best_ask, -best_ask_volume))

            #         best_bid = min(order_depth.buy_orders.keys())
            #         best_bid_volume = order_depth.buy_orders[best_bid]
            #         price_chg_bid = best_bid - previous_state_mid_price[product]

            #         print("best_bid =", best_bid)

            #         if best_bid > acceptable_price:
            #             print("SELL", product, str(best_bid_volume) + "x", best_bid)
            #             orders.append(Order(product, best_bid, -best_bid_volume))


            #     result[product] = orders                
            #     previous_state_mid_price[product] = mid_price


        print("Trader result:",result)
        my_pnl_value = np.sum([ my_pnl[key] * mid_prices[key] for key in my_pnl])
        print("PNL at end of turn:", my_pnl, "value = ", my_pnl_value)

        return result