from rich.table import Table

def trade_tracking_table(balance_info, open_trades):
    """
    Create a rich table for trade tracking information.
    """
    table = Table(title="Trade Tracking System", expand=True)
    
    table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
    table.add_column("Value", justify="right", style="green")

    # Balance Information
    table.add_row("Balance", balance_info.get("balance", "N/A"))
    table.add_row("Unrealized P/L", balance_info.get("unrealizedPL", "N/A"))
    table.add_row("Margin Used", balance_info.get("marginUsed", "N/A"))
    table.add_row("Margin Available", balance_info.get("marginAvailable", "N/A"))

    table.add_section()

    # Open Trades
    table.add_column("Open Trades", justify="left", style="cyan", no_wrap=True)
    if open_trades:
        for trade in open_trades:
            table.add_row(f"Instrument: {trade['instrument']}")
            table.add_row(f"Units: {trade['currentUnits']}")
            table.add_row(f"Open Price: {trade['price']}")
            table.add_row(f"Unrealized P/L: {trade['unrealizedPL']}")
    else:
        table.add_row("No open trades.")

    return table
