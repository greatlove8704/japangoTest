import pandas as pd
from pathlib import Path

class DataAligner:
    def __init__(self, raw_data_dir: str):
        self.data_dir = Path(raw_data_dir)
    
    def _clean_text(self, text: str) -> str:
        return text.strip()
    
    def _read_and_clean_file(self, filepath: Path, convert_float: bool = False) -> list:
        with open(filepath, 'r', encoding='utf-8') as f:
            if convert_float:
                return [float(self._clean_text(line)) for line in f]
            return [self._clean_text(line) for line in f]
    
    def align_and_save(self):
        # Read and clean files
        ja_lines = self._read_and_clean_file(self.data_dir / "combinedja.ja-vi.ja")
        vi_lines = self._read_and_clean_file(self.data_dir / "combinedvi.ja-vi.vi")
        scores = self._read_and_clean_file(self.data_dir / "combinedscores.ja-vi.scores", 
                                         convert_float=True)
        
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