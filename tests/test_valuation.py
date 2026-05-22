import pytest
import sys
import os
from unittest.mock import patch
from io import StringIO

# Add scripts directory to path to import valuation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))
from valuation import calculate_dcf, format_markdown_table, main

def test_standard_dcf_calculation():
    """
    Test standard DCF calculations against mathematically pre-computed inputs.
    FCF0 = 100, g1 = 10%, n1 = 3, g2 = 3%, WACC = 8%, debt = 500, shares = 100.
    """
    results = calculate_dcf(
        fcf=100.0,
        g1=0.10,
        n1=3,
        g2=0.03,
        wacc=0.08,
        shares=100.0,
        debt=500.0
    )
    
    # Expected FCFs:
    # Year 1: 110.0
    # Year 2: 121.0
    # Year 3: 133.1
    assert pytest.approx(results["projected_fcfs"][0], 1e-4) == 110.0
    assert pytest.approx(results["projected_fcfs"][1], 1e-4) == 121.0
    assert pytest.approx(results["projected_fcfs"][2], 1e-4) == 133.1
    
    # Expected Discount Factors:
    # DF1 = 1 / 1.08 = 0.925926
    # DF2 = 1 / 1.1664 = 0.857339
    # DF3 = 1 / 1.259712 = 0.793832
    assert pytest.approx(results["discount_factors"][0], 1e-4) == 1 / 1.08
    assert pytest.approx(results["discount_factors"][1], 1e-4) == 1 / 1.1664
    assert pytest.approx(results["discount_factors"][2], 1e-4) == 1 / 1.259712
    
    # Expected PV of FCFs:
    # PV1 = 110 * 0.9259259 = 101.85185
    # PV2 = 121 * 0.8573388 = 103.73800
    # PV3 = 133.1 * 0.7938322 = 105.65907
    # PV Stage 1 sum = 311.24892
    assert pytest.approx(results["pv_fcfs"][0], 1e-4) == 101.85185
    assert pytest.approx(results["pv_fcfs"][1], 1e-4) == 103.73800
    assert pytest.approx(results["pv_fcfs"][2], 1e-4) == 105.65907
    assert pytest.approx(results["pv_stage1_sum"], 1e-4) == 311.24892
    
    # Terminal Value:
    # TV = (133.1 * 1.03) / (0.08 - 0.03) = 137.093 / 0.05 = 2741.86
    # PV(TV) = 2741.86 / 1.259712 = 2176.57508
    assert pytest.approx(results["terminal_value"], 1e-4) == 2741.86
    assert pytest.approx(results["pv_terminal_value"], 1e-4) == 2176.57508
    
    # Enterprise Value:
    # EV = 311.24892 + 2176.57508 = 2487.824
    assert pytest.approx(results["enterprise_value"], 1e-4) == 2487.824
    
    # Equity Value = EV - Debt = 2487.824 - 500 = 1987.824
    assert pytest.approx(results["equity_value"], 1e-4) == 1987.824
    
    # Value Per Share = 1987.824 / 100 = 19.87824
    assert pytest.approx(results["value_per_share"], 1e-4) == 19.87824

def test_invalid_wacc_edge_case():
    """Verify that WACC <= 0 raises ValueError."""
    with pytest.raises(ValueError) as excinfo:
        calculate_dcf(fcf=100.0, g1=0.10, n1=5, g2=0.03, wacc=0.0)
    assert "WACC must be strictly positive" in str(excinfo.value)
    
    with pytest.raises(ValueError) as excinfo:
        calculate_dcf(fcf=100.0, g1=0.10, n1=5, g2=0.03, wacc=-0.05)
    assert "WACC must be strictly positive" in str(excinfo.value)

def test_wacc_less_than_or_equal_to_g2_edge_case():
    """Verify that WACC <= g2 raises ValueError to prevent division by zero or negative TV."""
    with pytest.raises(ValueError) as excinfo:
        calculate_dcf(fcf=100.0, g1=0.10, n1=5, g2=0.08, wacc=0.08)
    assert "must be strictly greater than terminal growth rate g2" in str(excinfo.value)
    
    with pytest.raises(ValueError) as excinfo:
        calculate_dcf(fcf=100.0, g1=0.10, n1=5, g2=0.09, wacc=0.08)
    assert "must be strictly greater than terminal growth rate g2" in str(excinfo.value)

def test_invalid_n1_years():
    """Verify that n1 <= 0 raises ValueError."""
    with pytest.raises(ValueError) as excinfo:
        calculate_dcf(fcf=100.0, g1=0.10, n1=0, g2=0.03, wacc=0.08)
    assert "Stage 1 years (n1) must be a positive integer" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        calculate_dcf(fcf=100.0, g1=0.10, n1=-5, g2=0.03, wacc=0.08)
    assert "Stage 1 years (n1) must be a positive integer" in str(excinfo.value)

def test_negative_shares_error():
    """Verify that negative shares raises ValueError."""
    with pytest.raises(ValueError) as excinfo:
        calculate_dcf(fcf=100.0, g1=0.10, n1=5, g2=0.03, wacc=0.08, shares=-50)
    assert "Shares outstanding cannot be negative" in str(excinfo.value)

def test_zero_or_negative_growth():
    """Verify zero or negative growth calculates correctly and does not crash."""
    results = calculate_dcf(
        fcf=100.0,
        g1=-0.05,  # Declining growth
        n1=3,
        g2=0.0,    # Zero terminal growth
        wacc=0.08
    )
    # Projections:
    # Year 1: 95.0
    # Year 2: 90.25
    # Year 3: 85.7375
    assert results["projected_fcfs"][0] == 95.0
    assert results["projected_fcfs"][1] == 90.25
    assert results["projected_fcfs"][2] == 85.7375
    
    # TV = (85.7375 * 1.0) / 0.08 = 1071.71875
    assert pytest.approx(results["terminal_value"], 1e-4) == 1071.71875

def test_negative_fcf_warning():
    """Verify negative FCF outputs a warning to stderr but calculates properly."""
    stderr_capture = StringIO()
    with patch("sys.stderr", new=stderr_capture):
        results = calculate_dcf(
            fcf=-100.0,
            g1=0.05,
            n1=3,
            g2=0.02,
            wacc=0.08
        )
    assert "WARNING: Starting FCF is negative" in stderr_capture.getvalue()
    assert results["projected_fcfs"][0] < 0
    assert results["enterprise_value"] < 0

def test_zero_shares_warning():
    """Verify zero shares prints warning and returns None for value_per_share."""
    stderr_capture = StringIO()
    with patch("sys.stderr", new=stderr_capture):
        results = calculate_dcf(
            fcf=100.0,
            g1=0.10,
            n1=5,
            g2=0.03,
            wacc=0.08,
            shares=0,
            debt=500.0
        )
    assert "WARNING: Shares outstanding is zero; per-share valuation skipped" in stderr_capture.getvalue()
    assert results["value_per_share"] is None

def test_format_markdown_table():
    """Verify standard formatting of markdown projection tables."""
    results = {
        "projected_fcfs": [110.0, 121.0],
        "discount_factors": [0.925926, 0.857339],
        "pv_fcfs": [101.85, 103.74]
    }
    table = format_markdown_table(2, results)
    assert "| Year 01 | $110.00 | 0.925926 | $101.85 |" in table
    assert "| Year 02 | $121.00 | 0.857339 | $103.74 |" in table

def test_cli_execution_success():
    """Verify that CLI behaves correctly with valid inputs."""
    test_args = [
        "valuation.py",
        "--fcf", "100.0",
        "--g1", "0.10",
        "--n1", "3",
        "--g2", "0.03",
        "--wacc", "0.08",
        "--shares", "100",
        "--debt", "500"
    ]
    with patch("sys.argv", test_args), patch("sys.stdout", new=StringIO()) as stdout_capture:
        main()
    
    output = stdout_capture.getvalue()
    assert "# Discounted Cash Flow (DCF) Valuation Report" in output
    assert "PV of Stage 1 Cash Flows" in output
    assert "Enterprise Value (EV)" in output
    assert "Implied Value Per Share" in output

def test_cli_execution_failure():
    """Verify that CLI exits with code 1 upon experiencing ValueError."""
    test_args = [
        "valuation.py",
        "--fcf", "100.0",
        "--g1", "0.10",
        "--n1", "3",
        "--g2", "0.09",  # g2 > wacc, invalid!
        "--wacc", "0.08"
    ]
    with patch("sys.argv", test_args), patch("sys.stderr", new=StringIO()) as stderr_capture:
        with pytest.raises(SystemExit) as exit_info:
            main()
    assert exit_info.value.code == 1
    assert "VALUATION ERROR:" in stderr_capture.getvalue()
