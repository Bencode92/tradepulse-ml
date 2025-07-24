"""
Commodity correlation mapping for TradePulse ML
Maps commodity codes to label indices for multi-label classification
"""

COMMODITY_CODES = [
    "AE:DIAMONDS", "AE:PETROLEUM_CRUDE", "AR:CORN", "AU:ALUMINIUM_ORE", 
    "AU:COAL", "AU:IRON_ORE", "AU:NATGAS", "AU:WHEAT", "AU:ZINC_ORE", 
    "BE:ZINC_METAL", "BO:ZINC_ORE", "BR:COFFEE", "BR:CORN", "BR:IRON_ORE", 
    "BR:MEAT", "BR:SOYBEAN", "BR:SUGAR", "CA:ALUMINIUM_METAL", 
    "CA:NICKEL_METAL", "CA:WHEAT", "CD:COPPER_REFINED", "CH:GOLD", 
    "CH:PHARMACEUTICALS", "CL:COPPER_ORE", "CL:COPPER_REFINED", 
    "CL:COPPER_UNREFINED", "CN:APPAREL", "CN:CARBON", "CN:CHEMICALS_MISC", 
    "CN:CHEMICALS_ORGANIC", "CN:ELECTRICAL_MACHINERY", "CN:FURNITURE", 
    "CN:MACHINERY", "CN:NICKEL_METAL", "CN:OPTICAL_INSTRUMENTS", "CN:PAPER", 
    "CN:PLASTICS", "CN:RARE_GASES", "CN:RUBBER", "CN:SHIPS", "CN:SILVER", 
    "CN:TOYS", "CN:VEHICLES", "CN:WOOD", "DE:AIRCRAFT", "DE:CHEMICALS_MISC", 
    "DE:COCOA", "DE:FINANCIAL_SERVICES", "DE:OPTICAL_INSTRUMENTS", "DE:PAPER", 
    "DE:PHARMACEUTICALS", "DE:RARE_GASES", "DE:VEHICLES", "FI:NICKEL_ORE", 
    "FR:AIRCRAFT", "FR:BEVERAGES", "FR:COSMETICS", "FR:ELECTRICITY", 
    "FR:FINANCIAL_SERVICES", "GB:GOLD", "GB:PLATINUM", "GB:SILVER", 
    "GN:ALUMINIUM_ORE", "HK:DIAMONDS", "HK:ELECTRICAL_MACHINERY", "HK:SILVER", 
    "ID:COAL", "ID:FERROALLOYS", "ID:PALM_OIL", "ID:TIN", "IN:DIAMONDS", 
    "IN:IT_SERVICES", "IN:RICE", "KR:SHIPS", "KZ:URANIUM", 
    "LU:FINANCIAL_SERVICES", "MX:LEAD_ORE", "MX:PRECIOUS_METALS_ORE", 
    "MY:PALM_OIL", "NA:URANIUM", "NG:URANIUM", "NL:COCOA", "NO:FISH", 
    "NO:NATGAS", "NO:NICKEL_METAL", "PK:RICE", "PE:COPPER_ORE", 
    "PE:LEAD_ORE", "PE:PRECIOUS_METALS_ORE", "PE:TIN", "PE:ZINC_ORE", 
    "PH:NICKEL_ORE", "QA:NATGAS", "RU:COAL", "RU:PETROLEUM_CRUDE", 
    "RU:PRECIOUS_METALS_ORE", "RU:WHEAT", "SE:COPPER_UNREFINED", 
    "SG:FINANCIAL_SERVICES", "TH:RICE", "UA:CORN", "US:CHEMICALS_MISC", 
    "US:CHEMICALS_ORGANIC", "US:CORN", "US:DIAMONDS", "US:EDIBLE_FRUITS", 
    "US:LEAD_ORE", "US:MEAT", "US:NATGAS", "US:OPTICAL_INSTRUMENTS", 
    "US:PETROLEUM_CRUDE", "US:PETROLEUM_REFINED", "US:PHARMACEUTICALS", 
    "US:PLASTICS", "US:PLATINUM", "US:RARE_GASES", "US:SOYBEAN", 
    "US:TRAVEL", "US:WHEAT", "US:ZINC_ORE", "VN:FOOTWEAR", "VN:RICE", 
    "ZA:FERROALLOYS", "ZA:PLATINUM", "ZM:COPPER_UNREFINED"
]

# Create reverse mapping for quick lookup
COMMODITY_TO_INDEX = {code: idx for idx, code in enumerate(COMMODITY_CODES)}

def correlations_to_labels(correlation_string):
    """
    Convert correlation string to binary label array
    
    Args:
        correlation_string: String like "US:WHEAT,CN:STEEL,AU:IRON_ORE"
        
    Returns:
        List of 0s and 1s matching COMMODITY_CODES indices
    """
    if not correlation_string or correlation_string == 'nan':
        return [0] * len(COMMODITY_CODES)
    
    correlations = [c.strip() for c in correlation_string.split(',') if c.strip()]
    labels = [0] * len(COMMODITY_CODES)
    
    for correlation in correlations:
        if correlation in COMMODITY_TO_INDEX:
            labels[COMMODITY_TO_INDEX[correlation]] = 1
    
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
