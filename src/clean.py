import pandas as pd
from pathlib import Path

class DataAligner:
    def __init__(self, raw_data_dir: str):
        self.data_dir = Path(raw_data_dir)
    
    def align_and_save(self):
        # Read files as-is
        with open(self.data_dir / "combinedja.ja-vi.ja", 'r', encoding='utf-8') as f:
            ja_lines = [line.strip() for line in f]
        
        with open(self.data_dir / "combinedvi.ja-vi.vi", 'r', encoding='utf-8') as f:
            vi_lines = [line.strip() for line in f]
            
        with open(self.data_dir / "combinedscores.ja-vi.scores", 'r', encoding='utf-8') as f:
            scores = [float(line.strip()) for line in f]
        
        # Create + save DataFrame
        df = pd.DataFrame({
            'japanese': ja_lines,
            'vietnamese': vi_lines,
            'score': scores
        })
        
        output_path = Path("data/aligned")
        output_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path / "parallel.csv", index=False, encoding='utf-8')

def main():
    aligner = DataAligner("data/raw")
    aligner.align_and_save()

if __name__ == "__main__":
    main()