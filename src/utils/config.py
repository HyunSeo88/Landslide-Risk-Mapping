"""
Configuration management using YAML files
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Data configuration"""
    root: str = "D:/Landslide/data"
    static_features: str = "processed/gyeongnam/static_features.csv"
    ldaps_stats: str = "processed/gyeongnam/LDAPS/statistics/ldaps_slope_statistics.csv"
    insar_timeseries: Optional[str] = None
    wildfire_decay: Optional[str] = None
    labels: str = "processed/gyeongnam/labels/landslide_labels.csv"
    graph_structure: str = "processed/gyeongnam/graph/env_similarity_graph.pkl"

    # Split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    type: str = "GNN_RNN_Hybrid"

    # GNN settings
    gnn_type: str = "GAT"  # GAT, GraphSAGE, GCN
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 2
    gnn_heads: int = 4
    gnn_dropout: float = 0.3

    # RNN settings
    rnn_type: str = "LSTM"  # LSTM, GRU
    rnn_input_dim: int = 6  # Number of dynamic features
    rnn_hidden_dim: int = 64
    rnn_num_layers: int = 2
    rnn_dropout: float = 0.3

    # Fusion settings
    fusion_hidden_dims: list = field(default_factory=lambda: [128, 64])
    fusion_dropout: float = 0.3


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    optimizer: str = "Adam"

    # Scheduler
    scheduler_type: str = "ReduceLROnPlateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001

    # Device
    device: str = "cuda"
    num_workers: int = 4


@dataclass
class Config:
    """Main configuration container"""
    project_name: str = "Landslide Risk Prediction"
    experiment_name: str = "baseline"
    seed: int = 42

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Paths
    output_dir: str = "D:/Landslide/experiments"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    viz_dir: str = "visualizations"


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to YAML config file

    Returns:
        Config object
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # Create nested config objects
    config = Config(
        project_name=config_dict.get('project_name', 'Landslide Risk Prediction'),
        experiment_name=config_dict.get('experiment_name', 'baseline'),
        seed=config_dict.get('seed', 42)
    )

    # Data config
    if 'data' in config_dict:
        config.data = DataConfig(**config_dict['data'])

    # Model config
    if 'model' in config_dict:
        model_dict = config_dict['model'].copy()
        if 'gnn' in model_dict:
            gnn = model_dict.pop('gnn')
            model_dict.update({f'gnn_{k}': v for k, v in gnn.items()})
        if 'rnn' in model_dict:
            rnn = model_dict.pop('rnn')
            model_dict.update({f'rnn_{k}': v for k, v in rnn.items()})
        if 'fusion' in model_dict:
            fusion = model_dict.pop('fusion')
            model_dict.update({f'fusion_{k}': v for k, v in fusion.items()})
        config.model = ModelConfig(**model_dict)

    # Training config
    if 'training' in config_dict:
        training_dict = config_dict['training'].copy()
        if 'scheduler' in training_dict:
            scheduler = training_dict.pop('scheduler')
            training_dict.update({f'scheduler_{k}': v for k, v in scheduler.items()})
        if 'early_stopping' in training_dict:
            early_stopping = training_dict.pop('early_stopping')
            training_dict.update({f'early_stopping_{k}': v for k, v in early_stopping.items()})
        config.training = TrainingConfig(**training_dict)

    # Output paths
    if 'output' in config_dict:
        output = config_dict['output']
        config.output_dir = output.get('experiments', config.output_dir)
        config.checkpoint_dir = output.get('checkpoints', config.checkpoint_dir)
        config.log_dir = output.get('logs', config.log_dir)
        config.viz_dir = output.get('visualizations', config.viz_dir)

    return config


def save_config(config: Config, save_path: str):
    """
    Save configuration to YAML file

    Args:
        config: Config object
        save_path: Path to save YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict
    config_dict = {
        'project_name': config.project_name,
        'experiment_name': config.experiment_name,
        'seed': config.seed,
        'data': config.data.__dict__,
        'model': {
            'type': config.model.type,
            'gnn': {
                'type': config.model.gnn_type,
                'hidden_dim': config.model.gnn_hidden_dim,
                'num_layers': config.model.gnn_num_layers,
                'heads': config.model.gnn_heads,
                'dropout': config.model.gnn_dropout,
            },
            'rnn': {
                'type': config.model.rnn_type,
                'input_dim': config.model.rnn_input_dim,
                'hidden_dim': config.model.rnn_hidden_dim,
                'num_layers': config.model.rnn_num_layers,
                'dropout': config.model.rnn_dropout,
            },
            'fusion': {
                'hidden_dims': config.model.fusion_hidden_dims,
                'dropout': config.model.fusion_dropout,
            }
        },
        'training': {
            'batch_size': config.training.batch_size,
            'num_epochs': config.training.num_epochs,
            'learning_rate': config.training.learning_rate,
            'weight_decay': config.training.weight_decay,
            'optimizer': config.training.optimizer,
            'scheduler': {
                'type': config.training.scheduler_type,
                'patience': config.training.scheduler_patience,
                'factor': config.training.scheduler_factor,
            },
            'early_stopping': {
                'patience': config.training.early_stopping_patience,
                'min_delta': config.training.early_stopping_min_delta,
            },
            'device': config.training.device,
            'num_workers': config.training.num_workers,
        },
        'output': {
            'experiments': config.output_dir,
            'checkpoints': config.checkpoint_dir,
            'logs': config.log_dir,
            'visualizations': config.viz_dir,
        }
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
