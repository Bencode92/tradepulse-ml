"""
Commodity correlation mapping for TradePulse ML
Maps commodity codes to label indices for multi-label classification
"""
import math

COMMODITY_CODES = [
    "AU:IRON_ORE", "AU:WHEAT", "AU:NATGAS", "BR:COFFEE", "BR:SOYBEAN", 
    "BR:SUGAR", "BR:IRON_ORE", "BR:MEAT", "CA:WHEAT", "CA:NICKEL_METAL", 
    "CA:ALUMINIUM_METAL", "CL:COPPER_ORE", "CL:COPPER_REFINED", "CN:APPAREL", 
    "CN:MACHINERY", "CN:ELECTRICAL_MACHINERY", "CN:SILVER", "CN:NICKEL_METAL", 
    "CN:CHEMICALS_ORGANIC", "CN:PLASTICS", "CN:VEHICLES", "CN:OPTICAL_INSTRUMENTS", 
    "CD:COPPER_REFINED", "FR:AIRCRAFT", "FR:ELECTRICITY", "FR:BEVERAGES", 
    "FR:FINANCIAL_SERVICES", "DE:COCOA", "DE:PHARMACEUTICALS", "DE:VEHICLES", 
    "DE:AIRCRAFT", "DE:OPTICAL_INSTRUMENTS", "HK:SILVER", "HK:ELECTRICAL_MACHINERY", 
    "IN:IT_SERVICES", "ID:PALM_OIL", "LU:FINANCIAL_SERVICES", "MY:PALM_OIL", 
    "NL:COCOA", "NO:NATGAS", "NO:NICKEL_METAL", "NO:FISH", "PE:COPPER_ORE", 
    "RU:PETROLEUM_CRUDE", "SG:FINANCIAL_SERVICES", "CH:GOLD", "CH:PHARMACEUTICALS", 
    "AE:PETROLEUM_CRUDE", "GB:SILVER", "GB:GOLD", "US:SOYBEAN", "US:WHEAT", 
    "US:PETROLEUM_CRUDE", "US:NATGAS", "US:MEAT", "US:CHEMICALS_ORGANIC", 
    "US:PHARMACEUTICALS", "US:PLASTICS", "US:OPTICAL_INSTRUMENTS", "US:TRAVEL", 
    "KZ:URANIUM"
]

# Create reverse mapping for quick lookup
COMMODITY_TO_INDEX = {code: idx for idx, code in enumerate(COMMODITY_CODES)}

def correlations_to_labels(correlation_string):
    """
    Convert correlation string to binary label array
    
    Args:
        correlation_string: String like "US:WHEAT,CN:STEEL,AU:IRON_ORE" or "US:WHEAT;CN:STEEL;AU:IRON_ORE"
        
    Returns:
        List of 0.0s and 1.0s matching COMMODITY_CODES indices (floats for BCELoss)
    """
    # Handle NaN and None values
    if correlation_string is None or (
        isinstance(correlation_string, float) and math.isnan(correlation_string)
    ):
        return [0.0] * len(COMMODITY_CODES)
    
    # Convert to string if not already
    if not isinstance(correlation_string, str):
        correlation_string = str(correlation_string)
    
    # Handle empty strings and 'nan' string
    if correlation_string.strip().lower() in {"", "nan"}:
        return [0.0] * len(COMMODITY_CODES)
    
    # Support both , and ; as separators
    if ';' in correlation_string:
        correlations = [c.strip() for c in correlation_string.split(';') if c.strip()]
    else:
        correlations = [c.strip() for c in correlation_string.split(',') if c.strip()]
    
    labels = [0.0] * len(COMMODITY_CODES)
    
    for correlation in correlations:
        if correlation in COMMODITY_TO_INDEX:
            labels[COMMODITY_TO_INDEX[correlation]] = 1.0
    
    return labels

def labels_to_correlations(labels):
    """
    Convert binary label array back to correlation string
    
    Args:
        labels: List of 0s and 1s or probabilities
        
    Returns:
        String like "US:WHEAT,CN:STEEL"
    """
    threshold = 0.5 if any(0 < l < 1 for l in labels) else 0.5
    
    correlations = [
        COMMODITY_CODES[i] 
        for i, label in enumerate(labels) 
        if label > threshold
    ]
    
    return ','.join(correlations) if correlations else ''

def get_commodity_name(code):
    """Extract commodity name from code"""
    if ':' in code:
        return code.split(':')[1]
    return code

def get_country_code(code):
    """Extract country code from commodity code"""
    if ':' in code:
        return code.split(':')[0]
    return ''

# Total number of unique commodity labels
NUM_CORRELATION_LABELS = len(COMMODITY_CODES)

print(f"✅ Loaded {NUM_CORRELATION_LABELS} commodity codes for correlation mapping")
