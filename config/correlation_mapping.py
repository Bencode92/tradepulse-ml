"""
Correlation mapping for TradePulse commodity detection
Maps country:commodity codes to indices for ML model
"""

COMMODITY_CODES = [
    "AE:DIAMONDS", "AE:PETROLEUM_CRUDE", "AR:CORN", "AU:ALUMINIUM_ORE", "AU:COAL", 
    "AU:IRON_ORE", "AU:NATGAS", "AU:WHEAT", "AU:ZINC_ORE", "BE:ZINC_METAL", 
    "BO:ZINC_ORE", "BR:COFFEE", "BR:CORN", "BR:IRON_ORE", "BR:MEAT", 
    "BR:SOYBEAN", "BR:SUGAR", "CA:ALUMINIUM_METAL", "CA:NICKEL_METAL", "CA:WHEAT", 
    "CD:COPPER_REFINED", "CH:GOLD", "CH:PHARMACEUTICALS", "CL:COPPER_ORE", "CL:COPPER_REFINED", 
    "CL:COPPER_UNREFINED", "CN:APPAREL", "CN:CARBON", "CN:CHEMICALS_MISC", "CN:CHEMICALS_ORGANIC", 
    "CN:ELECTRICAL_MACHINERY", "CN:FURNITURE", "CN:MACHINERY", "CN:NICKEL_METAL", "CN:OPTICAL_INSTRUMENTS", 
    "CN:PAPER", "CN:PLASTICS", "CN:RARE_GASES", "CN:RUBBER", "CN:SHIPS", 
    "CN:SILVER", "CN:TOYS", "CN:VEHICLES", "CN:WOOD", "DE:AIRCRAFT", 
    "DE:CHEMICALS_MISC", "DE:COCOA", "DE:FINANCIAL_SERVICES", "DE:OPTICAL_INSTRUMENTS", "DE:PAPER", 
    "DE:PHARMACEUTICALS", "DE:RARE_GASES", "DE:VEHICLES", "FI:NICKEL_ORE", "FR:AIRCRAFT", 
    "FR:BEVERAGES", "FR:COSMETICS", "FR:ELECTRICITY", "FR:FINANCIAL_SERVICES", "GB:GOLD", 
    "GB:PLATINUM", "GB:SILVER", "GN:ALUMINIUM_ORE", "HK:DIAMONDS", "HK:ELECTRICAL_MACHINERY", 
    "HK:SILVER", "ID:COAL", "ID:FERROALLOYS", "ID:PALM_OIL", "ID:TIN", 
    "IN:DIAMONDS", "IN:IT_SERVICES", "IN:RICE", "KR:SHIPS", "KZ:URANIUM", 
    "LU:FINANCIAL_SERVICES", "MX:LEAD_ORE", "MX:PRECIOUS_METALS_ORE", "MY:PALM_OIL", "NA:URANIUM", 
    "NG:URANIUM", "NL:COCOA", "NO:FISH", "NO:NATGAS", "NO:NICKEL_METAL", 
    "PK:RICE", "PE:COPPER_ORE", "PE:LEAD_ORE", "PE:PRECIOUS_METALS_ORE", "PE:TIN", 
    "PE:ZINC_ORE", "PH:NICKEL_ORE", "QA:NATGAS", "RU:COAL", "RU:PETROLEUM_CRUDE", 
    "RU:PRECIOUS_METALS_ORE", "RU:WHEAT", "SE:COPPER_UNREFINED", "SG:FINANCIAL_SERVICES", "TH:RICE", 
    "UA:CORN", "US:CHEMICALS_MISC", "US:CHEMICALS_ORGANIC", "US:CORN", "US:DIAMONDS", 
    "US:EDIBLE_FRUITS", "US:LEAD_ORE", "US:MEAT", "US:NATGAS", "US:OPTICAL_INSTRUMENTS", 
    "US:PETROLEUM_CRUDE", "US:PETROLEUM_REFINED", "US:PHARMACEUTICALS", "US:PLASTICS", "US:PLATINUM", 
    "US:RARE_GASES", "US:SOYBEAN", "US:TRAVEL", "US:WHEAT", "US:ZINC_ORE", 
    "VN:FOOTWEAR", "VN:RICE", "ZA:FERROALLOYS", "ZA:PLATINUM", "ZM:COPPER_UNREFINED"
]

# Create reverse mapping for quick lookup
COMMODITY_INDEX_MAP = {code: idx for idx, code in enumerate(COMMODITY_CODES)}

def correlations_to_labels(correlation_string):
    """
    Convert correlation string to binary label array
    
    Args:
        correlation_string: String like "US:WHEAT,CN:STEEL,AU:IRON_ORE" or empty
        
    Returns:
        List of 0s and 1s matching COMMODITY_CODES indices
    """
    if not correlation_string or correlation_string.strip() == "":
        return [0] * len(COMMODITY_CODES)
    
    correlations = [c.strip() for c in correlation_string.split(',')]
    labels = [0] * len(COMMODITY_CODES)
    
    for corr in correlations:
        if corr in COMMODITY_INDEX_MAP:
            labels[COMMODITY_INDEX_MAP[corr]] = 1
    
    return labels

def labels_to_correlations(labels, threshold=0.5):
    """
    Convert binary label array back to correlation string
    
    Args:
        labels: List of probabilities or binary values
        threshold: Minimum score to consider a correlation active
        
    Returns:
        String like "US:WHEAT,CN:STEEL"
    """
    active_correlations = []
    
    for idx, score in enumerate(labels):
        if score > threshold:
            active_correlations.append(COMMODITY_CODES[idx])
    
    return ','.join(active_correlations) if active_correlations else ""

# Commodity categories for grouping
COMMODITY_CATEGORIES = {
    "ENERGY": ["PETROLEUM_CRUDE", "PETROLEUM_REFINED", "NATGAS", "COAL", "URANIUM"],
    "METALS": ["GOLD", "SILVER", "PLATINUM", "COPPER", "IRON_ORE", "ALUMINIUM", "ZINC", "NICKEL", "TIN", "LEAD"],
    "AGRICULTURE": ["WHEAT", "CORN", "RICE", "SOYBEAN", "SUGAR", "COFFEE", "PALM_OIL", "MEAT", "FISH"],
    "INDUSTRIAL": ["STEEL", "CHEMICALS", "PLASTICS", "RUBBER", "PAPER", "WOOD"],
    "TECH": ["RARE_GASES", "ELECTRICAL_MACHINERY", "OPTICAL_INSTRUMENTS", "SEMICONDUCTORS"],
    "SERVICES": ["FINANCIAL_SERVICES", "IT_SERVICES", "TRAVEL"]
}
