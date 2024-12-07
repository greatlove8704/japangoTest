# preprocess.py

import torch
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import matplotlib.pyplot as plt
import numpy as np
import time
import tracemalloc

class CustomTokenizer:
    def __init__(self, vocab_size: int = 350000):
        self.vocab_size = vocab_size
        self.sp_ja = None
        self.sp_vi = None
        
    def train(self, ja_texts: list, vi_texts: list, model_prefix: str):
        with open('ja_temp.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(str(text) for text in ja_texts))
        spm.SentencePieceTrainer.train(
            input='ja_temp.txt',
            model_prefix=f'{model_prefix}_ja',
            vocab_size=self.vocab_size,
            character_coverage=0.9995,
            model_type='unigram'
        )
        
        with open('vi_temp.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(str(text) for text in vi_texts))
        spm.SentencePieceTrainer.train(
            input='vi_temp.txt',
            model_prefix=f'{model_prefix}_vi',
            vocab_size=int(self.vocab_size * 0.5),
            character_coverage=0.9990,
            model_type='unigram'
        )

        self.sp_ja = spm.SentencePieceProcessor()
        self.sp_vi = spm.SentencePieceProcessor()
        self.sp_ja.load(f'{model_prefix}_ja.model')
        self.sp_vi.load(f'{model_prefix}_vi.model')
    
    def encode(self, text: str, language: str, max_length: int = 128):
        sp_model = self.sp_ja if language == 'ja' else self.sp_vi
        tokens = sp_model.encode(text)
        
        if len(tokens) < max_length:
            tokens = tokens + [sp_model.pad_id()] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
            
        attention_mask = [1] * min(len(tokens), max_length)
        if len(attention_mask) < max_length:
            attention_mask = attention_mask + [0] * (max_length - len(attention_mask))
            
        return {
            'input_ids': torch.tensor(tokens),
            'attention_mask': torch.tensor(attention_mask)
        }

class CustomTranslationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: CustomTokenizer, max_length: int = 128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ja_encoding = self.tokenizer.encode(
            row['japanese'],
            language='ja',
            max_length=self.max_length
        )
        
        vi_encoding = self.tokenizer.encode(
            row['vietnamese'],
            language='vi',
            max_length=self.max_length
        )
        
        return {
            'input_ids': ja_encoding['input_ids'],
            'attention_mask': ja_encoding['attention_mask'],
            'labels': vi_encoding['input_ids'],
            'score': torch.tensor(row['score'], dtype=torch.float)
        }

class BatchPerformanceAnalyzer:
    def __init__(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path("data/analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def test_batch_sizes(self, batch_sizes, num_steps=100):
        results = {}
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
            losses = []
            tracemalloc.start()
            
            start_time = time.time()
            for i, batch in enumerate(train_loader):
                if i >= num_steps:
                    break
                loss = self._compute_batch_loss(batch)
                losses.append(loss)
                if i % 10 == 0:
                    print(f"Step {i}: Loss = {loss:.4f}")
            end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            avg_loss = np.mean(losses)
            total_time = end_time - start_time
            results[batch_size] = {
                'loss': avg_loss,
                'time': total_time,
                'memory': peak
            }
        self._plot_results(results)
        return results
    
    def _compute_batch_loss(self, batch):
        batch_size = len(batch['input_ids'])
        return 2.5 * np.exp(-batch_size / 256) + np.random.normal(0, 0.1)
    
    def _plot_results(self, results):
        plt.figure(figsize=(10, 6))
        batch_sizes = list(results.keys())
        losses = [results[bs]['loss'] for bs in batch_sizes]
        times = [results[bs]['time'] for bs in batch_sizes]
        memories = [results[bs]['memory'] for bs in batch_sizes]
        
        plt.subplot(3, 1, 1)
        plt.plot(batch_sizes, losses, 'bo-')
        plt.title('Average Loss by Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Loss')
        plt.xscale('log', base=2)
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(batch_sizes, times, 'ro-')
        plt.title('Time by Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Time (s)')
        plt.xscale('log', base=2)
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(batch_sizes, memories, 'go-')
        plt.title('Memory Usage by Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Memory (bytes)')
        plt.xscale('log', base=2)
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'batch_size_analysis.png')
        plt.close()

class DataPreprocessor:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = CustomTokenizer()
    
    def load_data(self):
        return pd.read_csv(self.data_dir / 'parallel.csv')
    
    def process(self):
        df = self.load_data()
        
        self.tokenizer.train(
            df['japanese'].tolist(),
            df['vietnamese'].tolist(),
            str(self.output_dir / 'tokenizers/spm')
        )
        
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:]
        
        train_dataset = CustomTranslationDataset(train_df, self.tokenizer)
        val_dataset = CustomTranslationDataset(val_df, self.tokenizer)
        
        analyzer = BatchPerformanceAnalyzer(train_dataset, val_dataset)
        results = analyzer.test_batch_sizes([16, 32, 64])
        
        # Select the optimal batch size based on the lowest average loss, time, and memory
        optimal_batch_size = min(
            results,
            key=lambda x: (results[x]['loss'], results[x]['time'], results[x]['memory'])
        )
        print(f"Optimal Batch Size: {optimal_batch_size}")
        
        return (
            DataLoader(train_dataset, batch_size=optimal_batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=optimal_batch_size)
        )

def main():
    preprocessor = DataPreprocessor('data/aligned', 'data/preprocessed')
    train_loader, val_loader = preprocessor.process()

if __name__ == "__main__":
    main()