import logging
import os
import time
import numpy as np
from decimal import Decimal
from typing import Dict, List, Deque, Optional
from collections import deque

from pydantic import Field

from hummingbot.client.config.config_data_types import BaseClientModel, ClientFieldData
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent, TradeType
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class EnhancedPMMConfig(BaseClientModel):
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    exchange: str = Field("binance_paper_trade", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Exchange where the bot will trade"))
    trading_pair: str = Field("ETH-USDT", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Trading pair in which the bot will place orders"))
    order_amount: Decimal = Field(0.01, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Order amount (denominated in base asset)"))
    base_bid_spread: Decimal = Field(0.001, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Base bid order spread (in percent)"))
    base_ask_spread: Decimal = Field(0.001, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Base ask order spread (in percent)"))
    order_refresh_time: int = Field(20, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Order refresh time (in seconds)"))
    price_type: str = Field("mid", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Price type to use (mid or last)"))
    volatility_adjustment: bool = Field(True, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Enable volatility-based spread adjustment"))
    volatility_adjustment_multiplier: Decimal = Field(Decimal("1.5"), client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Multiplier for volatility-based spread adjustment"))
    trend_following: bool = Field(True, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Enable trend-following for spread adjustments"))
    inventory_target_base_pct: Decimal = Field(Decimal("0.5"), client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Target base asset percentage of total inventory value (0-1)"))
    inventory_range_multiplier: Decimal = Field(Decimal("5.0"), client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Multiplier for inventory skew adjustment impact"))
    volatility_window: int = Field(20, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Number of price samples to calculate volatility"))
    trend_window: int = Field(50, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Number of price samples to determine trend"))
    max_spread: Decimal = Field(Decimal("0.01"), client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Maximum spread (in percent)"))
    min_spread: Decimal = Field(Decimal("0.0001"), client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Minimum spread (in percent)"))
    dynamic_order_sizing: bool = Field(True, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Enable dynamic order sizing based on volatility"))


class EnhancedPMM(ScriptStrategyBase):
    """
    Enhanced Pure Market Making Strategy
    
    This strategy enhances the basic PMM with:
    1. Volatility-based spread adjustments
    2. Trend analysis for adaptive spread sizing
    3. Inventory management to maintain balanced risk
    4. Dynamic order sizing based on market conditions
    """

    create_timestamp = 0
    price_source = PriceType.MidPrice

    @classmethod
    def init_markets(cls, config: EnhancedPMMConfig):
        cls.markets = {config.exchange: {config.trading_pair}}
        cls.price_source = PriceType.LastTrade if config.price_type == "last" else PriceType.MidPrice

    def __init__(self, connectors: Dict[str, ConnectorBase], config: EnhancedPMMConfig):
        super().__init__(connectors)
        self.config = config
        
        # Price history for calculations
        self.price_samples: Deque[Decimal] = deque(maxlen=max(config.volatility_window, config.trend_window))
        
        # Performance tracking
        self.total_filled_buys: Decimal = Decimal("0")
        self.total_filled_sells: Decimal = Decimal("0")
        self.buy_fill_sum: Decimal = Decimal("0")
        self.sell_fill_sum: Decimal = Decimal("0")
        
        # Last calculated values
        self.last_volatility: Optional[Decimal] = None
        self.last_trend: Optional[Decimal] = None
        
        # Sampling
        self.last_sample_timestamp = 0
        self.sample_interval = 2  # Sample price every 2 seconds
        
        self.log_with_clock(logging.INFO, "Enhanced PMM strategy initialized")

    def on_tick(self):
        """Main logic executed on each clock tick"""
        # Sample price for calculations
        self._collect_price_sample()
        
        # Create and place orders if it's time
        if self.create_timestamp <= self.current_timestamp:
            self.cancel_all_orders()
            
            # Calculate market metrics
            volatility = self._calculate_volatility()
            trend = self._calculate_trend()
            inventory_skew = self._calculate_inventory_skew()
            
            # Log the metrics
            self._log_metrics(volatility, trend, inventory_skew)
            
            # Create order proposals with adjusted spreads and sizes
            proposal: List[OrderCandidate] = self.create_proposal(volatility, trend, inventory_skew)
            proposal_adjusted: List[OrderCandidate] = self.adjust_proposal_to_budget(proposal)
            
            if proposal_adjusted:
                self.place_orders(proposal_adjusted)
            else:
                self.log_with_clock(logging.WARNING, "No orders to place after budget adjustment")
            
            self.create_timestamp = self.config.order_refresh_time + self.current_timestamp

    def _collect_price_sample(self):
        """Collect price samples for volatility and trend calculations"""
        current_time = time.time()
        if current_time - self.last_sample_timestamp >= self.sample_interval:
            price = self.connectors[self.config.exchange].get_price_by_type(
                self.config.trading_pair, self.price_source)
            if price:
                self.price_samples.append(price)
                self.last_sample_timestamp = current_time

    def _calculate_volatility(self) -> Decimal:
        """Calculate price volatility using standard deviation"""
        if len(self.price_samples) >= self.config.volatility_window:
            # Get most recent samples for the volatility window
            samples = list(self.price_samples)[-self.config.volatility_window:]
            # Calculate normalized standard deviation (coefficient of variation)
            mean_price = sum(samples) / len(samples)
            if mean_price == Decimal("0"):
                return Decimal("0")
            
            squared_deviations = [(price - mean_price) ** 2 for price in samples]
            variance = sum(squared_deviations) / len(squared_deviations)
            std_dev = Decimal(variance).sqrt()
            volatility = std_dev / mean_price
            self.last_volatility = volatility
            return volatility
        
        return self.last_volatility if self.last_volatility is not None else Decimal("0")

    def _calculate_trend(self) -> Decimal:
        """Calculate market trend using EMA gradient"""
        if len(self.price_samples) < self.config.trend_window:
            return Decimal("0")
        
        # Use recent prices for trend calculation
        recent_prices = list(self.price_samples)[-self.config.trend_window:]
        
        # Calculate simple moving average over different periods
        short_window = self.config.trend_window // 4
        short_sma = sum(recent_prices[-short_window:]) / min(short_window, len(recent_prices))
        long_sma = sum(recent_prices) / len(recent_prices)
        
        # Normalize trend between -1 and 1
        if long_sma == Decimal("0"):
            return Decimal("0")
        
        # Trend calculated as relative difference between short and long SMA
        trend = (short_sma - long_sma) / long_sma
        self.last_trend = trend
        return trend

    def _calculate_inventory_skew(self) -> Decimal:
        """
        Calculate inventory skew to balance portfolio
        Returns a value between -1 and 1:
        - Positive: We have excess base asset (should sell more/buy less)
        - Negative: We have excess quote asset (should buy more/sell less)
        - Zero: Perfect balance according to target
        """
        connector = self.connectors[self.config.exchange]
        base_asset, quote_asset = self.config.trading_pair.split("-")
        
        # Get balances
        base_balance = connector.get_available_balance(base_asset)
        quote_balance = connector.get_available_balance(quote_asset)
        
        # Get current mid price for conversion
        mid_price = connector.get_price_by_type(self.config.trading_pair, self.price_source)
        
        if mid_price == Decimal("0") or (base_balance == Decimal("0") and quote_balance == Decimal("0")):
            return Decimal("0")
        
        # Calculate total portfolio value in quote asset
        total_value_in_quote = base_balance * mid_price + quote_balance
        
        # Calculate the current base asset ratio
        if total_value_in_quote == Decimal("0"):
            return Decimal("0")
        
        current_base_ratio = (base_balance * mid_price) / total_value_in_quote
        
        # Calculate skew based on deviation from target ratio
        target = self.config.inventory_target_base_pct
        inventory_skew = (current_base_ratio - target) / target
        
        # Normalize to -1 to 1 range with config multiplier
        max_skew = Decimal("1")
        normalized_skew = max(min(inventory_skew, max_skew), -max_skew)
        
        return normalized_skew

    def create_proposal(self, 
                        volatility: Decimal, 
                        trend: Decimal, 
                        inventory_skew: Decimal) -> List[OrderCandidate]:
        """Create buy and sell orders with dynamic spreads based on market conditions"""
        ref_price = self.connectors[self.config.exchange].get_price_by_type(
            self.config.trading_pair, self.price_source)
        
        # Base spreads
        base_bid_spread = self.config.base_bid_spread
        base_ask_spread = self.config.base_ask_spread
        
        # --- Spread Adjustments ---
        
        # 1. Volatility adjustment (higher volatility = wider spreads)
        volatility_adjustment = Decimal("0")
        if self.config.volatility_adjustment and volatility > Decimal("0"):
            volatility_adjustment = volatility * self.config.volatility_adjustment_multiplier
        
        # 2. Trend following adjustment (strong trend = wider spread in trend direction)
        trend_adjustment_bid = Decimal("0")
        trend_adjustment_ask = Decimal("0")
        if self.config.trend_following:
            # Positive trend (price going up) - adjust ask higher, bid less
            # Negative trend (price going down) - adjust bid lower, ask less
            trend_factor = abs(trend) * Decimal("2")  # Amplify the trend effect
            if trend > Decimal("0"):  # Uptrend - widen ask more
                trend_adjustment_ask = trend_factor
                trend_adjustment_bid = -trend_factor * Decimal("0.5")  # Less adjustment for bids in uptrend
            else:  # Downtrend - widen bid more
                trend_adjustment_bid = trend_factor
                trend_adjustment_ask = -trend_factor * Decimal("0.5")  # Less adjustment for asks in downtrend
        
        # 3. Inventory skew adjustment (imbalanced inventory = adjusted spreads to rebalance)
        inventory_adjustment_bid = Decimal("0")
        inventory_adjustment_ask = Decimal("0")
        if abs(inventory_skew) > Decimal("0.01"):  # Only apply if skew is significant
            skew_impact = inventory_skew * self.config.inventory_range_multiplier
            # Positive skew (excess base) - decrease bid spread, increase ask spread
            # Negative skew (excess quote) - increase bid spread, decrease ask spread
            inventory_adjustment_bid = -skew_impact  # Negative skew makes this positive (wider bid)
            inventory_adjustment_ask = skew_impact   # Positive skew makes this positive (wider ask)
        
        # Total spread calculations with min/max limits
        bid_spread = max(
            min(
                base_bid_spread + volatility_adjustment + trend_adjustment_bid + inventory_adjustment_bid,
                self.config.max_spread
            ),
            self.config.min_spread
        )
        
        ask_spread = max(
            min(
                base_ask_spread + volatility_adjustment + trend_adjustment_ask + inventory_adjustment_ask,
                self.config.max_spread
            ),
            self.config.min_spread
        )
        
        # Calculate final prices
        buy_price = ref_price * (Decimal("1") - bid_spread)
        sell_price = ref_price * (Decimal("1") + ask_spread)
        
        # Dynamic order sizing based on volatility (optional)
        base_amount = Decimal(self.config.order_amount)
        buy_amount = base_amount
        sell_amount = base_amount
        
        if self.config.dynamic_order_sizing:
            # In high volatility, use smaller sizes
            if volatility > Decimal("0"):
                vol_factor = Decimal("1") - (volatility * Decimal("5"))  # Reduce size based on volatility
                vol_factor = max(vol_factor, Decimal("0.2"))  # Don't go below 20% of base size
                
                # Adjust for inventory skew - put larger orders on the side that rebalances inventory
                if inventory_skew > Decimal("0"):  # Excess base, increase buy size
                    buy_amount = base_amount * vol_factor
                    sell_amount = base_amount * min(Decimal("1.5"), vol_factor + abs(inventory_skew))
                else:  # Excess quote, increase sell size
                    buy_amount = base_amount * min(Decimal("1.5"), vol_factor + abs(inventory_skew))
                    sell_amount = base_amount * vol_factor
        
        # Create order candidates
        buy_order = OrderCandidate(
            trading_pair=self.config.trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.BUY,
            amount=buy_amount,
            price=buy_price
        )

        sell_order = OrderCandidate(
            trading_pair=self.config.trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.SELL,
            amount=sell_amount,
            price=sell_price
        )
        
        return [buy_order, sell_order]

    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        """Adjust order proposal to available budget"""
        return self.connectors[self.config.exchange].budget_checker.adjust_candidates(proposal, all_or_none=False)

    def place_orders(self, proposal: List[OrderCandidate]) -> None:
        """Place orders from the proposal"""
        for order in proposal:
            self.place_order(connector_name=self.config.exchange, order=order)

    def place_order(self, connector_name: str, order: OrderCandidate):
        """Place an individual order"""
        if order.order_side == TradeType.SELL:
            self.sell(
                connector_name=connector_name,
                trading_pair=order.trading_pair,
                amount=order.amount,
                order_type=order.order_type,
                price=order.price
            )
        elif order.order_side == TradeType.BUY:
            self.buy(
                connector_name=connector_name,
                trading_pair=order.trading_pair,
                amount=order.amount,
                order_type=order.order_type,
                price=order.price
            )

    def cancel_all_orders(self):
        """Cancel all active orders"""
        for order in self.get_active_orders(connector_name=self.config.exchange):
            self.cancel(self.config.exchange, order.trading_pair, order.client_order_id)

    def did_fill_order(self, event: OrderFilledEvent):
        """Handle order filled event and update fill statistics"""
        # Update fill statistics for performance tracking
        if event.trade_type == TradeType.BUY:
            self.total_filled_buys += event.amount
            self.buy_fill_sum += event.amount * event.price
        else:
            self.total_filled_sells += event.amount
            self.sell_fill_sum += event.amount * event.price
        
        # Log the fill
        msg = (f"{event.trade_type.name} {round(event.amount, 6)} {event.trading_pair} {self.config.exchange} "
               f"at {round(event.price, 6)}")
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)
        
        # Force order refresh after fill to quickly adapt to new market conditions
        self.create_timestamp = 0

    def _log_metrics(self, volatility: Decimal, trend: Decimal, inventory_skew: Decimal):
        """Log current market metrics for monitoring"""
        metrics_msg = (
            f"Market metrics - Volatility: {round(volatility, 6)}, "
            f"Trend: {round(trend, 6)}, "
            f"Inventory Skew: {round(inventory_skew, 6)}"
        )
        self.log_with_clock(logging.INFO, metrics_msg)
        
        # Log performance stats occasionally
        if self.total_filled_buys > 0 and self.total_filled_sells > 0:
            avg_buy_price = self.buy_fill_sum / self.total_filled_buys if self.total_filled_buys > 0 else Decimal("0")
            avg_sell_price = self.sell_fill_sum / self.total_filled_sells if self.total_filled_sells > 0 else Decimal("0")
            
            if avg_buy_price > 0:
                profit_pct = ((avg_sell_price / avg_buy_price) - Decimal("1")) * Decimal("100")
                performance_msg = (
                    f"Performance - Avg Buy: {round(avg_buy_price, 6)}, "
                    f"Avg Sell: {round(avg_sell_price, 6)}, "
                    f"Profit %: {round(profit_pct, 4)}%"
                )
                self.log_with_clock(logging.INFO, performance_msg) 