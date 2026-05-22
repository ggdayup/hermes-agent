#!/usr/bin/env python3
"""
Enterprise-grade Two-Stage Discounted Cash Flow (DCF) valuation model.
Calculates projected FCFs, Terminal Value (TV), Present Values (PV) of FCFs and TV, 
and Enterprise Value (EV). Formats results as markdown and handles financial edge cases.
"""

import argparse
import sys
from typing import Dict, Any, List, Optional

def calculate_dcf(
    fcf: float,
    g1: float,
    n1: int,
    g2: float,
    wacc: float,
    shares: Optional[float] = None,
    debt: Optional[float] = None
) -> Dict[str, Any]:
    """
    Computes a two-stage DCF valuation model.
    
    Args:
        fcf: Base year Free Cash Flow (FCF_0).
        g1: Stage 1 growth rate (short-term growth).
        n1: Number of years in Stage 1.
        g2: Stage 2 growth rate (terminal growth).
        wacc: Weighted Average Cost of Capital (discount rate).
        shares: Optional number of shares outstanding.
        debt: Optional net debt (Total Debt - Cash).
        
    Returns:
        A dictionary containing projected cash flows, terminal values, present values,
        enterprise value, equity value, and implied per-share value.
    """
    # Validation of inputs and edge cases
    if wacc <= 0:
        raise ValueError(f"WACC must be strictly positive (received: {wacc:.2%}).")
        
    if n1 <= 0:
        raise ValueError(f"Stage 1 years (n1) must be a positive integer (received: {n1}).")
        
    if wacc <= g2:
        raise ValueError(
            f"WACC ({wacc:.2%}) must be strictly greater than terminal growth rate g2 ({g2:.2%}) "
            f"to prevent an infinite or negative Terminal Value."
        )
        
    if shares is not None and shares < 0:
        raise ValueError(f"Shares outstanding cannot be negative (received: {shares}).")

    # Issue warnings for unusual but mathematically valid scenarios
    if fcf < 0:
        print(
            f"WARNING: Starting FCF is negative ({fcf:,.2f}). This may lead to negative valuation metrics.",
            file=sys.stderr
        )

    # 1. Stage 1 Projections
    projected_fcfs: List[float] = []
    discount_factors: List[float] = []
    pv_fcfs: List[float] = []
    
    current_fcf = fcf
    for year in range(1, n1 + 1):
        current_fcf = current_fcf * (1 + g1)
        projected_fcfs.append(current_fcf)
        
        df = 1 / ((1 + wacc) ** year)
        discount_factors.append(df)
        
        pv = current_fcf * df
        pv_fcfs.append(pv)
        
    pv_stage1_sum = sum(pv_fcfs)
    
    # 2. Stage 2 (Terminal Value)
    last_stage1_fcf = projected_fcfs[-1]
    terminal_value = (last_stage1_fcf * (1 + g2)) / (wacc - g2)
    pv_terminal_value = terminal_value / ((1 + wacc) ** n1)
    
    # 3. Enterprise Value (EV)
    enterprise_value = pv_stage1_sum + pv_terminal_value
    
    # 4. Equity Value & Per-Share Value
    equity_value: Optional[float] = None
    value_per_share: Optional[float] = None
    
    if debt is not None:
        equity_value = enterprise_value - debt
        if shares is not None:
            if shares > 0:
                value_per_share = equity_value / shares
            else:
                print("WARNING: Shares outstanding is zero; per-share valuation skipped.", file=sys.stderr)
    elif shares is not None:
        # If shares are provided but debt is not, assume net debt = 0
        equity_value = enterprise_value
        if shares > 0:
            value_per_share = equity_value / shares
        else:
            print("WARNING: Shares outstanding is zero; per-share valuation skipped.", file=sys.stderr)

    return {
        "projected_fcfs": projected_fcfs,
        "discount_factors": discount_factors,
        "pv_fcfs": pv_fcfs,
        "pv_stage1_sum": pv_stage1_sum,
        "terminal_value": terminal_value,
        "pv_terminal_value": pv_terminal_value,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "value_per_share": value_per_share
    }

def format_markdown_table(n1: int, results: Dict[str, Any]) -> str:
    """Formats the year-by-year cash flows into a markdown table."""
    lines = [
        "| Year | Projected FCF | Discount Factor | Present Value |",
        "| :--- | :------------ | :-------------- | :------------ |"
    ]
    for year in range(1, n1 + 1):
        fcf = results["projected_fcfs"][year - 1]
        df = results["discount_factors"][year - 1]
        pv = results["pv_fcfs"][year - 1]
        lines.append(f"| Year {year:02d} | ${fcf:,.2f} | {df:.6f} | ${pv:,.2f} |")
    return "\n".join(lines)

def print_valuation_report(
    fcf: float,
    g1: float,
    n1: int,
    g2: float,
    wacc: float,
    shares: Optional[float],
    debt: Optional[float],
    results: Dict[str, Any]
) -> None:
    """Prints a beautiful enterprise-grade DCF valuation report."""
    print("# Discounted Cash Flow (DCF) Valuation Report")
    print("\n## Model Input Parameters")
    print(f"- **Base FCF ($FCF_0$):** ${fcf:,.2f}")
    print(f"- **Short-Term Growth Rate ($g_1$):** {g1:.2%}")
    print(f"- **Stage 1 Duration ($N_1$):** {n1} years")
    print(f"- **Terminal Growth Rate ($g_2$):** {g2:.2%}")
    print(f"- **Weighted Average Cost of Capital ($WACC$):** {wacc:.2%}")
    if debt is not None:
        print(f"- **Net Debt:** ${debt:,.2f}")
    if shares is not None:
        print(f"- **Shares Outstanding:** {shares:,.0f}")
        
    print("\n## Stage 1 Projections Table")
    print(format_markdown_table(n1, results))
    
    print("\n## Valuation Summary")
    print(f"- **PV of Stage 1 Cash Flows:** ${results['pv_stage1_sum']:,.2f}")
    print(f"- **Terminal Value (TV):** ${results['terminal_value']:,.2f}")
    print(f"- **PV of Terminal Value:** ${results['pv_terminal_value']:,.2f}")
    print(f"- **Enterprise Value (EV):** ${results['enterprise_value']:,.2f}")
    
    if results['equity_value'] is not None:
        print(f"- **Implied Equity Value:** ${results['equity_value']:,.2f}")
    if results['value_per_share'] is not None:
        print(f"- **Implied Value Per Share:** **${results['value_per_share']:,.2f}**")
    print("\n" + "="*50 + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enterprise-grade Two-Stage Discounted Cash Flow (DCF) Valuation Tool."
    )
    parser.add_argument("--fcf", type=float, required=True, help="Base year Free Cash Flow (FCF_0)")
    parser.add_argument("--g1", type=float, required=True, help="Stage 1 growth rate (e.g. 0.15 for 15%)")
    parser.add_argument("--n1", type=int, required=True, help="Stage 1 duration in years")
    parser.add_argument("--g2", type=float, required=True, help="Terminal growth rate (e.g. 0.03 for 3%)")
    parser.add_argument("--wacc", type=float, required=True, help="Weighted Average Cost of Capital (e.g. 0.08 for 8%)")
    parser.add_argument("--shares", type=float, default=None, help="Optional shares outstanding for per-share value")
    parser.add_argument("--debt", type=float, default=None, help="Optional Net Debt (Total Debt - Cash)")

    args = parser.parse_args()

    try:
        results = calculate_dcf(
            fcf=args.fcf,
            g1=args.g1,
            n1=args.n1,
            g2=args.g2,
            wacc=args.wacc,
            shares=args.shares,
            debt=args.debt
        )
        print_valuation_report(
            fcf=args.fcf,
            g1=args.g1,
            n1=args.n1,
            g2=args.g2,
            wacc=args.wacc,
            shares=args.shares,
            debt=args.debt,
            results=results
        )
    except ValueError as e:
        print(f"VALUATION ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
