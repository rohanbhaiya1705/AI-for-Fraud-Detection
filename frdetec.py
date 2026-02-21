"""
Data loading utilities for fraud detection.
Handles loading and initial validation of transaction data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import json


@dataclass
class DataSchema:
    """Defines expected schema for transaction data."""
    
    # Required columns
    transaction_id: str = "transaction_id"
    timestamp: str = "timestamp"
    amount: float = "amount"
    user_id: str = "user_id"
    merchant_id: str = "merchant_id"
    
    # Optional but recommended
    category: str = "category"
    location: str = "location"
    device_id: str = "device_id"
    ip_address: str = "ip_address"
    
    # Target column
    is_fraud: str = "is_fraud"
    
    # Transaction attributes (for feature engineering)
    card_present: str = "card_present"
    entry_mode: str = "entry_mode"
    currency: str = "currency"


class DataLoader:
    """
    Loads transaction data from various sources.
    Supports CSV, Parquet, and database connections.
    """
    
    def __init__(self, schema: Optional[DataSchema] = None):
        self.schema = schema or DataSchema()
        self.loaded_data: Optional[pd.DataFrame] = None
    
    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load transaction data from CSV file.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame with transaction data
        """
        df = pd.read_csv(file_path, **kwargs)
        self.loaded_data = self._validate_and_standardize(df)
        return self.loaded_data
    
    def load_parquet(self, file_path: str) -> pd.DataFrame:
        """Load transaction data from Parquet file."""
        df = pd.read_parquet(file_path)
        self.loaded_data = self._validate_and_standardize(df)
        return self.loaded_data
    
    def load_synthetic_data(
        self,
        n_transactions: int = 100000,
        fraud_ratio: float = 0.02,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Generate synthetic transaction data for development and testing.
        Creates realistic-looking transaction patterns with labeled fraud.
        
        Args:
            n_transactions: Number of transactions to generate
            fraud_ratio: Proportion of fraudulent transactions
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with synthetic transaction data
        """
        np.random.seed(seed)
        
        n_fraud = int(n_transactions * fraud_ratio)
        n_normal = n_transactions - n_fraud
        
        # Generate user IDs
        n_users = max(1000, n_transactions // 100)
        n_merchants = max(500, n_transactions // 200)
        
        # Normal transactions
        normal_data = {
            self.schema.transaction_id: [f"TXN{i:08d}" for i in range(n_normal)],
            self.schema.user_id: np.random.randint(1, n_users, n_normal),
            self.schema.merchant_id: np.random.randint(1, n_merchants, n_normal),
            self.schema.amount: np.abs(np.random.lognormal(4, 1.5, n_normal)),
            self.schema.timestamp: pd.date_range(
                "2024-01-01", periods=n_normal, freq="30s"
            ).astype(str),
            self.schema.category: np.random.choice(
                ["retail", "food", "travel", "entertainment", "utilities"], n_normal
            ),
            self.schema.location: np.random.choice(
                ["US", "UK", "EU", "CA", "AU"], n_normal
            ),
            self.schema.is_fraud: 0
        }
        
        # Fraud transactions (with suspicious patterns)
        fraud_data = {
            self.schema.transaction_id: [f"TXN{i:08d}" for i in range(n_normal, n_transactions)],
            self.schema.user_id: np.random.randint(1, n_users, n_fraud),
            self.schema.merchant_id: np.random.randint(1, n_merchants, n_fraud),
            self.schema.amount: np.abs(np.random.lognormal(6, 1.2, n_fraud)),
            self.schema.timestamp: pd.date_range(
                "2024-01-01", periods=n_fraud, freq="45s"
            ).astype(str),
            self.schema.category: np.random.choice(
                ["retail", "electronics", "jewelry"], n_fraud
            ),
            self.schema.location: np.random.choice(
                ["US", "UK", "RU", "CN"], n_fraud
            ),
            self.schema.is_fraud: 1
        }
        
        # Combine and shuffle
        df_normal = pd.DataFrame(normal_data)
        df_fraud = pd.DataFrame(fraud_data)
        df = pd.concat([df_normal, df_fraud], ignore_index=True)
        
        # Add additional features
        df["card_present"] = np.random.choice([True, False], n_transactions, p=[0.7, 0.3])
        df["entry_mode"] = np.random.choice(
            ["chip", "swipe", "online"], n_transactions, p=[0.4, 0.3, 0.3]
        )
        df["device_id"] = [f"DEV{np.random.randint(1, 1000)}" for _ in range(n_transactions)]
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        self.loaded_data = df
        return df
    
    def _validate_and_standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate data against schema and standardize column names.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Validated and standardized DataFrame
        """
        # Check required columns
        required_cols = [
            self.schema.transaction_id,
            self.schema.amount,
            self.schema.user_id,
        ]
        
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Standardize amount column
        if self.schema.amount not in df.columns and "amount" in df.columns:
            df = df.rename(columns={"amount": self.schema.amount})
        
        return df
    
    def split_data(
        self,
        test_size: float = 0.2,
        stratify: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.
        
        Args:
            test_size: Proportion of data for testing
            stratify: Whether to maintain class balance
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if self.loaded_data is None:
            raise ValueError("No data loaded. Call load_csv or load_synthetic_data first.")
        
        if stratify:
            return train_test_split(
                self.loaded_data,
                test_size=test_size,
                random_state=42,
                stratify=self.loaded_data[self.schema.is_fraud]
            )
        else:
            return train_test_split(
                self.loaded_data,
                test_size=test_size,
                random_state=42
            )


def train_test_split(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    stratify: Optional[pd.Series] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple train-test split with optional stratification."""
    n = len(df)
    test_n = int(n * test_size)
    
    indices = np.random.RandomState(random_state).permutation(n)
    test_indices = indices[:test_n]
    train_indices = indices[test_n:]
    
    if stratify is not None:
        # Ensure class balance in both sets
        train_idx, test_idx = [], []
        stratify_arr = stratify.values
        
        for idx in train_indices:
            train_idx.append(idx)
        for idx in test_indices:
            test_idx.append(idx)
        
        return df.iloc