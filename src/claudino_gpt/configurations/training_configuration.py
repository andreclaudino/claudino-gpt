from dataclasses import dataclass


@dataclass
class TrainingConfiguration:
    random_seed: int
    learning_rate: float
    weight_decay: float

    batch_size: int
    number_of_epochs: int
    data_loading_workers_count: int

    evaluation_frequency: int
    evaluation_iteration: int

    train_source_path: str
    validation_source_path: str

    checkpoint_frequency: int

    output_path: str

    total_models_to_keep: int
    
