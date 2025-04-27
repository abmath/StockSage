def format_currency(value):
    """
    Format a number as a currency with ₹ sign and commas
    
    Args:
        value (float): The value to format
        
    Returns:
        str: Formatted currency string
    """
    if value is None:
        return "N/A"
    
    try:
        return f"₹{float(value):,.2f}"
    except:
        return "N/A"

def format_percentage(value):
    """
    Format a number as a percentage with % sign
    
    Args:
        value (float): The value to format
        
    Returns:
        str: Formatted percentage string
    """
    if value is None:
        return "N/A"
    
    try:
        formatted = f"{float(value):+.2f}%"
        return formatted
    except:
        return "N/A"

def validate_portfolio_data(df):
    """
    Validate that the uploaded portfolio data has the required columns
    
    Args:
        df (DataFrame): The portfolio dataframe to validate
        
    Returns:
        dict: Validation result with 'valid' (bool) and 'message' (str) keys
    """
    # Required columns
    required_columns = ["Instrument", "Qty", "Avg cost"]
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return {
            "valid": False,
            "message": f"Missing required columns: {', '.join(missing_columns)}"
        }
    
    # Validate instrument column - should contain ticker symbols
    if df["Instrument"].isna().any():
        return {
            "valid": False,
            "message": "Instrument column contains empty values"
        }
    
    # Validate quantity column - should be numeric and positive
    try:
        qty = df["Qty"].astype(float)
        if (qty <= 0).any():
            return {
                "valid": False,
                "message": "Quantity values must be positive numbers"
            }
    except:
        return {
            "valid": False,
            "message": "Quantity column must contain numeric values"
        }
    
    # Validate average cost column - should be numeric and positive
    try:
        avg_cost = df["Avg cost"].astype(float)
        if (avg_cost <= 0).any():
            return {
                "valid": False,
                "message": "Average cost values must be positive numbers"
            }
    except:
        return {
            "valid": False,
            "message": "Average cost column must contain numeric values"
        }
    
    # All validations passed
    return {
        "valid": True,
        "message": "Data is valid"
    }
