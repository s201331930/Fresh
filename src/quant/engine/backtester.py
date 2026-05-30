"""Event-driven, long-only, leverage-free backtest engine.

Core guarantees enforced by construction:
    * LONG ONLY     -> we only ever buy and later sell; no negative share counts.
    * LEVERAGE FREE -> every purchase is gated by available cash, so aggregate
                       position value can never exceed equity.
    * RISK MANAGED  -> each open position carries an initial hard stop and may
                       carry a take-profit and/or a ratcheting trailing stop.

Execution conventions (to avoid look-ahead bias):
    * Strategy entry/exit signals are evaluated on the CLOSE of day *t* and
      executed at the OPEN of day *t+1*.
    * Stop-loss / take-profit / trailing exits are intrabar: checked against the
      HIGH/LOW of each bar while the position is open. When a bar could plausibly
      hit both the stop and the target, we conservatively assume the STOP fills
      first (worst case for the strategy).
    * Gaps through the stop are filled at the (worse) open price.

Frictions: per-side commission (bps) and adverse slippage (bps) on every fill.
Idle cash accrues the risk-free rate daily.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..data import PriceData


@dataclass
class RiskConfig:
    """Per-strategy risk / sizing parameters."""

    # Initial hard stop -- specify EITHER a percentage OR an ATR multiple.
    stop_loss_pct: Optional[float] = None
    stop_atr_mult: Optional[float] = None
    # Optional profit target.
    take_profit_pct: Optional[float] = None
    # Optional trailing stop -- percentage and/or ATR multiple (tighter wins).
    trailing_pct: Optional[float] = None
    atr_trailing_mult: Optional[float] = None
    # Time stop (max bars held).
    max_holding_days: Optional[int] = None
    # Sizing.
    sizing: str = "equal"          # "equal" | "risk"
    risk_per_trade: float = 0.01   # used when sizing == "risk"
    max_positions: int = 5


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series
    cash: pd.Series
    exposure: pd.Series           # invested fraction of equity (0..1)
    trades: List[dict] = field(default_factory=list)

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().fillna(0.0)


@dataclass
class _Position:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    stop: float
    take_profit: Optional[float]
    trailing_pct: Optional[float]
    trailing_atr: Optional[float]   # absolute dollar trail distance (mult*ATR)
    highest: float
    bars_held: int = 0


class Backtester:
    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        risk_free_rate: float = 0.0,
        commission_bps: float = 1.0,
        slippage_bps: float = 5.0,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.rf = float(risk_free_rate)
        self.commission = commission_bps / 10_000.0
        self.slippage = slippage_bps / 10_000.0

    # ------------------------------------------------------------------ #
    def run(
        self,
        data: PriceData,
        signals: Dict[str, pd.DataFrame],
        risk: RiskConfig,
        name: str = "strategy",
    ) -> BacktestResult:
        """Run the backtest.

        ``signals[symbol]`` must be a DataFrame aligned to that symbol's bars
        with boolean columns ``entry`` and ``exit`` and an optional ``atr``
        column (required when ATR-based stops/sizing are used).
        """
        symbols = [s for s in data.symbols if s in signals]
        if not symbols:
            raise ValueError(f"[{name}] no tradable symbols with signals")

        # Master calendar = union of all symbol dates.
        all_dates = sorted(set().union(*[set(data[s].index) for s in symbols]))
        calendar = pd.DatetimeIndex(all_dates)

        # Per-symbol aligned arrays for fast access.
        bars: Dict[str, pd.DataFrame] = {}
        sig: Dict[str, pd.DataFrame] = {}
        for s in symbols:
            b = data[s].reindex(calendar)
            bars[s] = b
            sg = signals[s].reindex(calendar)
            for col in ("entry", "exit"):
                if col not in sg:
                    sg[col] = False
            sg["entry"] = sg["entry"].fillna(False).astype(bool)
            sg["exit"] = sg["exit"].fillna(False).astype(bool)
            sig[s] = sg

        cash = self.initial_capital
        positions: Dict[str, _Position] = {}
        pending_entries: List[str] = []
        pending_exits: List[str] = []
        trades: List[dict] = []

        equity_curve = np.empty(len(calendar))
        cash_curve = np.empty(len(calendar))
        exposure_curve = np.empty(len(calendar))

        daily_rf = (1 + self.rf) ** (1 / 252) - 1 if self.rf else 0.0
        last_close: Dict[str, float] = {}

        for i, date in enumerate(calendar):
            # Idle cash accrues the risk-free rate.
            if daily_rf:
                cash *= 1 + daily_rf

            # 1) EXECUTE PENDING SIGNAL EXITS at today's open.
            for s in pending_exits:
                pos = positions.get(s)
                if pos is None:
                    continue
                px = bars[s]["open"].iat[i]
                if np.isnan(px):
                    continue
                cash += self._sell_proceeds(px, pos.shares)
                trades.append(self._close_trade(pos, date, self._fill_sell(px), "signal"))
                del positions[s]
            pending_exits = []

            # 2) EXECUTE PENDING ENTRIES at today's open.
            sleeve_equity = cash + self._positions_value(positions, bars, i, last_close)
            for s in pending_entries:
                if s in positions or len(positions) >= risk.max_positions:
                    continue
                row = bars[s].iloc[i]
                px = row["open"]
                if np.isnan(px):
                    continue
                atr = sig[s]["atr"].iat[i] if "atr" in sig[s] else np.nan
                pos = self._open_position(s, date, px, atr, sleeve_equity, cash, risk)
                if pos is None:
                    continue
                cost = self._buy_cost(self._fill_buy(px), pos.shares)
                if cost > cash + 1e-9:
                    continue  # leverage guard: never spend more than cash
                cash -= cost
                positions[s] = pos
            pending_entries = []

            # 3) INTRABAR RISK MANAGEMENT on open positions.
            for s in list(positions.keys()):
                pos = positions[s]
                row = bars[s].iloc[i]
                o, h, l = row["open"], row["high"], row["low"]
                if np.isnan(h) or np.isnan(l):
                    continue
                pos.bars_held += 1

                # Ratchet the trailing stop using the running high-water mark.
                pos.highest = max(pos.highest, h)
                trail_candidates = [pos.stop]
                if pos.trailing_pct is not None:
                    trail_candidates.append(pos.highest * (1 - pos.trailing_pct))
                if pos.trailing_atr is not None:
                    trail_candidates.append(pos.highest - pos.trailing_atr)
                pos.stop = max(trail_candidates)

                exit_price = None
                reason = None
                # Stop first (conservative). Gap through stop -> fill at open.
                if not np.isnan(o) and o <= pos.stop:
                    exit_price, reason = o, "stop"
                elif l <= pos.stop:
                    exit_price, reason = pos.stop, "stop"
                elif pos.take_profit is not None and h >= pos.take_profit:
                    exit_price = o if (not np.isnan(o) and o >= pos.take_profit) else pos.take_profit
                    reason = "target"
                elif risk.max_holding_days and pos.bars_held >= risk.max_holding_days:
                    cl = row["close"]
                    if not np.isnan(cl):
                        exit_price, reason = cl, "time"

                if exit_price is not None:
                    cash += self._sell_proceeds(self._fill_sell(exit_price), pos.shares)
                    trades.append(self._close_trade(pos, date, self._fill_sell(exit_price), reason))
                    del positions[s]

            # 4) MARK-TO-MARKET at the close.
            for s in symbols:
                c = bars[s]["close"].iat[i]
                if not np.isnan(c):
                    last_close[s] = c
            pos_val = self._positions_value(positions, bars, i, last_close)
            equity = cash + pos_val
            equity_curve[i] = equity
            cash_curve[i] = cash
            exposure_curve[i] = pos_val / equity if equity > 0 else 0.0

            # 5) GENERATE NEXT-DAY ORDERS from signals at the close.
            for s in symbols:
                row = sig[s].iloc[i]
                has_bar = not np.isnan(bars[s]["close"].iat[i])
                if not has_bar:
                    continue
                if bool(row["entry"]) and s not in positions:
                    pending_entries.append(s)
                if bool(row["exit"]) and s in positions:
                    pending_exits.append(s)

        equity = pd.Series(equity_curve, index=calendar, name=name)
        cash_s = pd.Series(cash_curve, index=calendar, name="cash")
        expo = pd.Series(exposure_curve, index=calendar, name="exposure")
        return BacktestResult(name=name, equity=equity, cash=cash_s, exposure=expo, trades=trades)

    # ------------------------------------------------------------------ #
    # Sizing / position helpers
    # ------------------------------------------------------------------ #
    def _open_position(
        self,
        symbol: str,
        date: pd.Timestamp,
        raw_price: float,
        atr: float,
        sleeve_equity: float,
        cash: float,
        risk: RiskConfig,
    ) -> Optional[_Position]:
        fill = self._fill_buy(raw_price)

        # Initial hard stop.
        if risk.stop_atr_mult is not None and not np.isnan(atr) and atr > 0:
            stop = fill - risk.stop_atr_mult * atr
        elif risk.stop_loss_pct is not None:
            stop = fill * (1 - risk.stop_loss_pct)
        else:
            stop = fill * (1 - 0.10)  # safety net: never run a stop-less book
        stop = max(stop, 0.0)

        # Position sizing.
        per_slot_value = sleeve_equity / max(1, risk.max_positions)
        if risk.sizing == "risk":
            stop_dist = max(fill - stop, 1e-9)
            risk_dollars = sleeve_equity * risk.risk_per_trade
            target_value = min(risk_dollars / stop_dist * fill, per_slot_value * 1.0)
        else:
            target_value = per_slot_value

        budget = min(target_value, cash / (1 + self.commission + self.slippage))
        shares = int(math.floor(budget / fill)) if fill > 0 else 0
        if shares <= 0:
            return None

        take_profit = fill * (1 + risk.take_profit_pct) if risk.take_profit_pct else None
        trailing_atr = (
            risk.atr_trailing_mult * atr
            if (risk.atr_trailing_mult is not None and not np.isnan(atr) and atr > 0)
            else None
        )
        return _Position(
            symbol=symbol,
            entry_date=date,
            entry_price=fill,
            shares=shares,
            stop=stop,
            take_profit=take_profit,
            trailing_pct=risk.trailing_pct,
            trailing_atr=trailing_atr,
            highest=fill,
        )

    @staticmethod
    def _positions_value(positions, bars, i, last_close) -> float:
        total = 0.0
        for s, pos in positions.items():
            c = bars[s]["close"].iat[i]
            price = c if not np.isnan(c) else last_close.get(s, pos.entry_price)
            total += price * pos.shares
        return total

    # --- fills & costs --- #
    def _fill_buy(self, price: float) -> float:
        return price * (1 + self.slippage)

    def _fill_sell(self, price: float) -> float:
        return price * (1 - self.slippage)

    def _buy_cost(self, fill_price: float, shares: int) -> float:
        notional = fill_price * shares
        return notional * (1 + self.commission)

    def _sell_proceeds(self, fill_price: float, shares: int) -> float:
        notional = fill_price * shares
        return notional * (1 - self.commission)

    def _close_trade(self, pos: _Position, date, exit_fill: float, reason: str) -> dict:
        gross_entry = pos.entry_price * pos.shares
        gross_exit = exit_fill * pos.shares
        commissions = (gross_entry + gross_exit) * self.commission
        pnl = gross_exit - gross_entry - commissions
        ret = (exit_fill / pos.entry_price - 1.0) if pos.entry_price else 0.0
        return {
            "symbol": pos.symbol,
            "entry_date": pos.entry_date,
            "exit_date": date,
            "entry_price": round(pos.entry_price, 4),
            "exit_price": round(exit_fill, 4),
            "shares": pos.shares,
            "pnl": float(pnl),
            "return_pct": float(ret),
            "holding_days": int(pos.bars_held),
            "exit_reason": reason,
        }
