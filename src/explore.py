import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import unicodedata

class DataExplorer:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df = pd.read_csv(
            self.data_path,
            header=0,
            dtype={
                'japanese': str,
                'vietnamese': str,
                'score': float
            },
            low_memory=False
        )
        # Add one line to remove any null values that might cause issues
        self.df = self.df.dropna(subset=['japanese', 'vietnamese'])
    
    def analyze_writing_systems(self, text: str) -> dict:
        counts = {'kanji': 0, 'hiragana': 0, 'katakana': 0, 'other': 0}
        for char in text:
            name = unicodedata.name(char, '')
            if 'CJK UNIFIED' in name:
                counts['kanji'] += 1
            elif 'HIRAGANA' in name:
                counts['hiragana'] += 1
            elif 'KATAKANA' in name:
                counts['katakana'] += 1
            else:
                counts['other'] += 1
        return counts
    
    def generate_plots(self):
        output_dir = Path("data/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Length distribution
        plt.figure(figsize=(12, 6))
        plt.hist([self.df['japanese'].str.len(), self.df['vietnamese'].str.len()], 
                 bins=50, label=['Japanese', 'Vietnamese'])
        plt.title('Sentence length distribution')
        plt.xlabel('Characters')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(output_dir / 'length_distribution.png')
        plt.close()

        # Length ratio distribution
        length_ratios = self.df['vietnamese'].str.len() / self.df['japanese'].str.len()
        plt.figure(figsize=(12, 6))
        plt.hist(length_ratios.clip(0, 5), bins=50)
        plt.title('Vietnamese/Japanese Length ratio distribution')
        plt.xlabel('Length Ratio (Vi/Ja)')
        plt.ylabel('Frequency')
        plt.savefig(output_dir / 'length_ratio.png')
        plt.close()

        # Japanese writing system distribution
        ja_writing = {k: 0 for k in ['kanji', 'hiragana', 'katakana', 'other']}
        sample_size = min(100000, len(self.df))
        for text in self.df['japanese'].sample(sample_size):
            counts = self.analyze_writing_systems(text)
            for k, v in counts.items():
                ja_writing[k] += v
                
        plt.figure(figsize=(10, 10))
        plt.pie(ja_writing.values(), labels=ja_writing.keys(), autopct='%1.1f%%')
        plt.title('Japanese writing system distribution')
        plt.savefig(output_dir / 'writing_distribution.png')
        plt.close()
    
    def generate_report(self):
        output_dir = Path("data/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate statistics
        ja_lengths = self.df['japanese'].str.len()
        vi_lengths = self.df['vietnamese'].str.len()
        
        # Calculate Vi/Ja character ratio
        total_ja_chars = ja_lengths.sum()
        total_vi_chars = vi_lengths.sum()
        char_ratio = total_vi_chars / total_ja_chars
        
        # Convert to numpy array for reliable percentile calculation
        ja_lengths_array = np.array(ja_lengths)
        vi_lengths_array = np.array(vi_lengths)
        
        with open(output_dir / "analysis_report.txt", 'w', encoding='utf-8') as f:
            f.write("* Translation dataset analysis *\n\n")
            f.write(f"Total pairs: {len(self.df):,}\n")
            f.write(f"Unique sentences: {self.df['japanese'].nunique():,} (Ja), "
                   f"{self.df['vietnamese'].nunique():,} (Vi)\n\n")
            
            f.write("Length Statistics:\n")
            f.write(f"Japanese: avg={ja_lengths.mean():.1f}, "
                   f"95th percentile={np.percentile(ja_lengths_array, 95):.0f}\n")
            f.write(f"Vietnamese: avg={vi_lengths.mean():.1f}, "
                   f"95th percentile={np.percentile(vi_lengths_array, 95):.0f}\n\n")
            
            f.write(f"Average Vi/Ja character ratio: {char_ratio:.2f}\n")

def main():
    explorer = DataExplorer("data/aligned/parallel.csv")
    explorer.generate_plots()
    explorer.generate_report()
    print("completed")

if __name__ == "__main__":
    main()