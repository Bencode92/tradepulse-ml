def _standardise(self, rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        text = row.get("text") or (
            f"{row.get('title', '')} {row.get('content', '')}"
        ).strip()
        
        # 🔧 FIX : Nettoyage robuste des labels
        if self.target_column == "importance":
            label_raw = row.get("importance", "")
        else:
            label_raw = (
                row.get("label") 
                or row.get("sentiment") 
                or row.get("impact") 
                or ""
            )
        
        # 🔧 FIX : Nettoyage strict
        if label_raw is None:
            label_raw = ""
        
        label = str(label_raw).strip().lower()
        
        # 🔧 FIX : Debug si label non reconnu
        if label and label not in self.LABEL_MAP:
            logger.warning(f"⚠️ Label non reconnu: '{label}' (raw: '{label_raw}') pour target_column='{self.target_column}'")
            logger.warning(f"Labels attendus: {list(self.LABEL_MAP.keys())}")
            continue
            
        if not text or not label:
            continue
            
        out.append({"text": text, "label": self.LABEL_MAP[label]})
    return out