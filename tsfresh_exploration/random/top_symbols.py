import json
from datetime import datetime
from pathlib import Path
import pandas as pd

class TopSymbolTracker:
    """Track and maintain list of most statistically extreme symbols."""
    
    def __init__(self, file_path='top_symbols.json'):
        self.file_path = file_path
        self.top_symbols = self._load_data()
        
        # Initialize if no existing data
        if not self.top_symbols:
            self.top_symbols = {
                'metadata': {
                    'last_updated': None,
                    'total_symbols_analyzed': 0
                },
                'by_category': {
                    'overall_score': [],
                    'volatility': [],
                    'momentum': [],
                    'volume_patterns': [],
                    'price_patterns': [],
                    'correlation_patterns': []
                },
                'symbols_history': {}  # Track all scores for each symbol
            }
    
    def _load_data(self):
        """Load existing top symbols data."""
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    
    def _save_data(self):
        """Save top symbols data with backup."""
        if Path(self.file_path).exists():
            backup_file = self.file_path + '.backup'
            try:
                import shutil
                shutil.copy2(self.file_path, backup_file)
            except Exception as e:
                print(f"Warning: Could not create backup: {str(e)}")
        
        self.top_symbols['metadata']['last_updated'] = datetime.now().isoformat()
        
        try:
            with open(self.file_path, 'w') as f:
                json.dump(self.top_symbols, f, indent=2)
        except Exception as e:
            print(f"Error saving top symbols: {str(e)}")
            if Path(backup_file).exists():
                shutil.copy2(backup_file, self.file_path)
                print("Restored from backup")
    
    def update(self, symbol: str, patterns: dict):
        """Update top symbols based on new pattern analysis."""
        self.top_symbols['metadata']['total_symbols_analyzed'] += 1
        
        # Extract scores for different categories
        scores = {
            'overall_score': patterns['score'],
            'volatility': self._calculate_volatility_score(patterns),
            'momentum': self._calculate_momentum_score(patterns),
            'volume_patterns': self._calculate_volume_score(patterns),
            'price_patterns': self._calculate_price_score(patterns),
            'correlation_patterns': self._calculate_correlation_score(patterns)
        }
        
        # Update symbol history
        timestamp = datetime.now().isoformat()
        if symbol not in self.top_symbols['symbols_history']:
            self.top_symbols['symbols_history'][symbol] = []
        
        self.top_symbols['symbols_history'][symbol].append({
            'timestamp': timestamp,
            'scores': scores,
            'patterns': {
                'major_patterns': patterns.get('major_patterns', []),
                'regime_changes': patterns.get('regime_changes', []),
                'correlations': patterns.get('correlations', []),
                'notes': patterns.get('notes', [])
            }
        })
        
        # Update top symbols for each category
        for category, score in scores.items():
            category_list = self.top_symbols['by_category'][category]
            
            # Add new entry
            entry = {
                'symbol': symbol,
                'score': score,
                'timestamp': timestamp
            }
            
            # Insert while maintaining sort
            category_list.append(entry)
            category_list.sort(key=lambda x: x['score'], reverse=True)
            
            # Only keep meaningful scores and use percentile threshold
            min_symbols = 50  # Wait for minimum sample size
            if self.top_symbols['metadata']['total_symbols_analyzed'] >= min_symbols:
                # Keep top 5% but require minimum scores
                cutoff = max(10, int(self.top_symbols['metadata']['total_symbols_analyzed'] * 0.05))
                filtered_list = [
                    entry for entry in category_list 
                    if entry['score'] > 0.1  # Minimum score threshold
                ]
                self.top_symbols['by_category'][category] = filtered_list[:cutoff]
            else:
                # During initial data collection, keep all scores for baseline
                self.top_symbols['by_category'][category] = category_list
        
        self._save_data()
    
    def _calculate_volatility_score(self, patterns: dict) -> float:
        """Calculate volatility score from patterns."""
        score = 0.0
        
        # Base score from pattern presence
        if any('volatility' in p.lower() for p in patterns.get('major_patterns', [])):
            score += 2.0
        if any('range' in p.lower() for p in patterns.get('major_patterns', [])):
            score += 2.0
        if any('structural breaks' in p.lower() for p in patterns.get('regime_changes', [])):
            score += 2.0
            
        # Add z-score contributions
        for note in patterns.get('notes', []):
            if 'z-score' in note:
                try:
                    zscore = float(note.split('z-score: ')[1].strip(')'))
                    score += min(1.0, abs(zscore) / 4)  # Up to 1.0 per strong z-score
                except:
                    continue
            
        return min(10.0, score)
    
    def _calculate_momentum_score(self, patterns: dict) -> float:
        """Calculate momentum score from patterns."""
        score = 0.0
        
        # Base score from pattern presence
        if any('trend' in p.lower() for p in patterns.get('major_patterns', [])):
            score += 2.0
        if any('momentum' in p.lower() for p in patterns.get('major_patterns', [])):
            score += 2.0
            
        # Add autocorrelation contributions
        autocorr_count = 0
        for note in patterns.get('notes', []):
            if 'autocorrelation' in note.lower():
                autocorr_count += 1
                try:
                    zscore = float(note.split('z-score: ')[1].strip(')'))
                    score += min(0.5, abs(zscore) / 4)  # Up to 0.5 per strong autocorrelation
                except:
                    continue
        
        if autocorr_count > 0:
            score += min(2.0, autocorr_count * 0.2)  # Up to 2.0 for multiple autocorrelations
            
        return min(10.0, score)
    
    def _calculate_volume_score(self, patterns: dict) -> float:
        """Calculate volume pattern score."""
        score = 0.0
        
        # Check for volume-related patterns
        if any('volume' in p.lower() for p in patterns.get('major_patterns', [])):
            score += 2.0
        if any('volume' in c.lower() for c in patterns.get('correlations', [])):
            score += 2.0
        if any('volume' in n.lower() for n in patterns.get('notes', [])):
            score += 2.0
            
        return min(10.0, score)
    
    def _calculate_price_score(self, patterns: dict) -> float:
        """Calculate price pattern score."""
        score = 0.0
        
        # Check for price-related patterns
        if any('price' in p.lower() for p in patterns.get('major_patterns', [])):
            score += 2.0
        if any('tail' in p.lower() for p in patterns.get('major_patterns', [])):
            score += 2.0
        if any('distribution' in p.lower() for p in patterns.get('major_patterns', [])):
            score += 2.0
            
        return min(10.0, score)
    
    def _calculate_correlation_score(self, patterns: dict) -> float:
        """Calculate correlation pattern score."""
        score = 0.0
        
        # Check for correlation-related patterns
        if patterns.get('correlations'):
            score += len(patterns['correlations']) * 2.0
            
        return min(10.0, score)
    
    def get_summary(self) -> dict:
        """Get summary of current top symbols."""
        summary = {
            'total_analyzed': self.top_symbols['metadata']['total_symbols_analyzed'],
            'last_updated': self.top_symbols['metadata']['last_updated'],
            'top_by_category': {}
        }
        
        for category, symbols in self.top_symbols['by_category'].items():
            if symbols:
                summary['top_by_category'][category] = [
                    {'symbol': s['symbol'], 'score': s['score']} 
                    for s in symbols[:5]  # Top 5 for each category
                ]
        
        return summary
    
    def get_symbol_history(self, symbol: str) -> list:
        """Get historical pattern data for a specific symbol."""
        return self.top_symbols['symbols_history'].get(symbol, [])
