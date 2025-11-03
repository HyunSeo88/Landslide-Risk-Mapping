"""
Dynamic Negative Sampler for Hierarchical Fusion Model

Generates negative samples dynamically each epoch from the full slope pool,
ensuring diverse training data and preventing overfitting.

Author: Landslide Risk Analysis Project
Date: 2025-01-26
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import random


class DynamicNegativeSampler:
    """
    Dynamic negative sampling strategy for landslide prediction.

    Key Features:
    - Samples from slopes that never had landslides (not in positive set)
    - 80% of negatives: dates matched to positive sample distribution
    - 20% of negatives: completely random dates
    - Generates new negative set each epoch for training diversity

    Example:
        >>> positive_df = pd.read_csv('positive_samples.csv')
        >>> sampler = DynamicNegativeSampler(
        ...     positive_samples_df=positive_df,
        ...     all_slope_ids=list(range(1, 87697)),
        ...     date_range=('2019-01-01', '2020-09-30'),
        ...     ratio=1.0,
        ...     random_date_ratio=0.2
        ... )
        >>> negative_df = sampler.generate_epoch_samples()
    """

    def __init__(
        self,
        positive_samples_df: pd.DataFrame,
        all_slope_ids: List[int],
        date_range: Tuple[str, str],
        ratio: float = 1.0,
        random_date_ratio: float = 0.2,
        seed: int = None,
        exclude_slope_ids: set = None
    ):
        """
        Initialize dynamic negative sampler.

        Args:
            positive_samples_df: DataFrame with columns [cat, event_date, label]
                                Contains positive samples for THIS training period
            all_slope_ids: List of all slope IDs in the study area
            date_range: Tuple of (start_date, end_date) for valid sampling dates
                       Format: 'YYYY-MM-DD'
            ratio: Negative:Positive ratio (default: 1.0 for balanced sampling)
            random_date_ratio: Fraction of negatives with random dates (default: 0.2)
            seed: Random seed for reproducibility (optional)
            exclude_slope_ids: Set of slope IDs to exclude from negative pool
                             (e.g., all historical landslide slopes)
        """
        self.positive_df = positive_samples_df
        self.ratio = ratio
        self.random_date_ratio = random_date_ratio

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Extract positive slope IDs from current training period
        self.positive_slope_ids = set(self.positive_df['cat'].unique())

        # Create negative pool
        # Exclude: (1) current period positives + (2) all historical positives
        if exclude_slope_ids is not None:
            slopes_to_exclude = exclude_slope_ids  # Use provided exclusion set
        else:
            slopes_to_exclude = self.positive_slope_ids  # Fallback to current positives only

        self.negative_pool = [sid for sid in all_slope_ids if sid not in slopes_to_exclude]

        print(f"\n=== Dynamic Negative Sampler Initialized ===")
        print(f"Total slopes: {len(all_slope_ids):,}")
        print(f"Positive slopes (current period): {len(self.positive_slope_ids):,}")
        if exclude_slope_ids is not None:
            print(f"Excluded slopes (all history): {len(exclude_slope_ids):,}")
        print(f"Negative pool: {len(self.negative_pool):,}")
        print(f"Ratio (neg:pos): {ratio}")
        print(f"Random date ratio: {random_date_ratio*100:.0f}%")

        # Validate negative pool
        if len(self.negative_pool) == 0:
            raise ValueError("Negative pool is empty! All slopes are in positive set.")

        # Parse date range
        self.start_date = pd.to_datetime(date_range[0])
        self.end_date = pd.to_datetime(date_range[1])

        # Generate all available dates
        self.all_dates = pd.date_range(self.start_date, self.end_date, freq='D')
        self.all_dates_str = [d.strftime('%Y-%m-%d') for d in self.all_dates]

        print(f"Date range: {date_range[0]} ~ {date_range[1]} ({len(self.all_dates)} days)")

        # Precompute positive date distribution (for weighted sampling)
        self.positive_date_counts = self.positive_df['event_date'].value_counts()
        self.positive_dates = self.positive_date_counts.index.tolist()
        self.positive_weights = self.positive_date_counts.values / self.positive_date_counts.sum()

        print(f"Positive unique dates: {len(self.positive_dates)}")
        print(f"Top 5 positive dates:")
        for date, count in self.positive_date_counts.head(5).items():
            print(f"  {date}: {count} samples")

    def generate_epoch_samples(self, epoch: int = None) -> pd.DataFrame:
        """
        Generate negative samples for one epoch.

        Args:
            epoch: Current epoch number (for logging, optional)

        Returns:
            negative_samples_df: DataFrame with columns [cat, event_date, label]
                                label is always 0 for negative samples
        """
        n_positive = len(self.positive_df)
        n_negative = int(n_positive * self.ratio)

        if n_negative > len(self.negative_pool):
            print(f"Warning: Requested {n_negative} negatives but pool has {len(self.negative_pool)}.")
            print(f"Sampling with replacement.")
            replacement = True
        else:
            replacement = False

        # Sample slopes from negative pool
        sampled_slope_ids = np.random.choice(
            self.negative_pool,
            size=n_negative,
            replace=replacement
        )

        # Determine date assignment strategy
        n_matched_dates = int(n_negative * (1 - self.random_date_ratio))
        n_random_dates = n_negative - n_matched_dates

        # 80%: Sample dates matching positive distribution
        matched_dates = np.random.choice(
            self.positive_dates,
            size=n_matched_dates,
            p=self.positive_weights,
            replace=True
        )

        # 20%: Sample completely random dates
        random_dates = np.random.choice(
            self.all_dates_str,
            size=n_random_dates,
            replace=True
        )

        # Combine dates and shuffle
        all_dates = np.concatenate([matched_dates, random_dates])
        np.random.shuffle(all_dates)

        # Create negative samples DataFrame
        negative_df = pd.DataFrame({
            'cat': sampled_slope_ids,
            'event_date': all_dates,
            'label': 0
        })

        if epoch is not None:
            print(f"\n[Epoch {epoch}] Generated {n_negative} negative samples:")
        else:
            print(f"\nGenerated {n_negative} negative samples:")

        print(f"  Matched dates (80%): {n_matched_dates}")
        print(f"  Random dates (20%): {n_random_dates}")
        print(f"  Unique slopes: {negative_df['cat'].nunique()}")
        print(f"  Unique dates: {negative_df['event_date'].nunique()}")

        return negative_df

    def get_combined_samples(self, epoch: int = None) -> pd.DataFrame:
        """
        Generate negative samples and combine with positive samples.

        Args:
            epoch: Current epoch number (for logging, optional)

        Returns:
            combined_df: DataFrame with all positive + negative samples, shuffled
        """
        negative_df = self.generate_epoch_samples(epoch)

        # Combine positive and negative
        combined_df = pd.concat([self.positive_df, negative_df], ignore_index=True)

        # Shuffle
        combined_df = combined_df.sample(frac=1.0, random_state=None).reset_index(drop=True)

        print(f"  Total combined samples: {len(combined_df)}")
        print(f"    Positive: {(combined_df['label']==1).sum()}")
        print(f"    Negative: {(combined_df['label']==0).sum()}")

        return combined_df
