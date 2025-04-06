# NPC-Group-submission-assignment

Problem: Static strategies fail in crypto’s volatile markets due to:   - Volatility traps (orders filled at bad prices).   - Trends causing adverse selection (buying dips/selling rallies).   - Inventory imbalances (overexposure pre-crash).   

Solution:  1. Volatility-Adjusted Spreads   - Mechanism: Uses `volatility_window` to widen 
spreads in chaos (protection) or tighten in calm 
(opportunity).   
2. Trend Alignment   - Mechanism: Detects momentum via `trend_window`, 
nudging spreads to follow market direction (e.g., tighter 
asks in uptrends).   
3. Inventory Rebalancing   - Mechanism: Adjusts spreads based on holdings vs. target 
(e.g., `inventory_target_base_pct`), incentivizing sales if 
overexposed.   
4. Dynamic Sizing (Optional): Reduces order size in high 
volatility.  
