import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from classifiers import RNNClassifier, NaiveClassifier
from visualization import plot_single_model_results, plot_model_comparison


def run_classification_experiments(
    data_dir: str = "../clean_data/airquality_prepared",
    output_dir: str = "rnn_classification_results_pytorch",
    sequence_length: int = 24,
    epochs: int = 50,
    device: str = None
):
    # save path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # load data
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)
    
    data_dir = Path(data_dir)
    X_all = pd.read_parquet(data_dir / "features" / "nn" / "X.parquet")
    y_all = pd.read_parquet(data_dir / "features" / "nn" / "y.parquet")
    splits = json.loads((data_dir / "splits.json").read_text())
    
    # extract data
    train_idx = splits['train']
    val_idx = splits['val']
    test_idx = splits['test']
    
    X_train = X_all.iloc[train_idx].values
    X_val = X_all.iloc[val_idx].values
    X_test = X_all.iloc[test_idx].values
    
    y_train = y_all['target_cls'].iloc[train_idx].values
    y_val = y_all['target_cls'].iloc[val_idx].values
    y_test = y_all['target_cls'].iloc[test_idx].values
    
    print(f"Train set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Check data
    assert not np.isnan(X_train).any() and not np.isnan(y_train).any(), "Training data contains NaN"
    assert not np.isnan(X_val).any() and not np.isnan(y_val).any(), "Validation data contains NaN"
    assert not np.isnan(X_test).any() and not np.isnan(y_test).any(), "Test data contains NaN"

    print("\nClass Distribution:")
    print(f"Train: {np.bincount(y_train.astype(int))} = {len(y_train)} samples")
    print(f"Val:   {np.bincount(y_val.astype(int))} = {len(y_val)} samples")
    print(f"Test:  {np.bincount(y_test.astype(int))} = {len(y_test)} samples")
    
    # validate scope
    label_check_idx = 0
    label_check_list = [("Train", y_train), ("Val", y_val), ("Test", y_test)]
    while label_check_idx < len(label_check_list):
        name, y_data = label_check_list[label_check_idx]
        unique_labels = np.unique(y_data.astype(int))
        assert np.all((unique_labels >= 0) & (unique_labels <= 2)), \
            f"{name} labels contain invalid values: {unique_labels}"
        label_check_idx += 1
    
    # Prepare
    horizons = [1, 6, 12, 24] 
    model_configs = {
        'LSTM': {'model_type': 'lstm', 'hidden_units': 64, 'num_layers': 2},
        'GRU': {'model_type': 'gru', 'hidden_units': 64, 'num_layers': 2}
    }
    
    all_results = {}  
    
    print("\n" + "="*60)
    print("Experiment Configuration")
    print("="*60)
    print(f"Sequence length: {sequence_length} hours")
    print(f"Forecast horizons: {horizons} hours")
    print(f"Models: {list(model_configs.keys())}")
    print(f"Device: {device if device else 'auto-detect'}")
    print(f"Max epochs: {epochs}")
    print(f"Batch size: 32")
    
    # base
    print("\n" + "="*60)
    print("Naive Baseline")
    print("="*60)
    
    naive = NaiveClassifier()
    naive_results = {}
    
    horizon_idx = 0
    while horizon_idx < len(horizons):
        horizon = horizons[horizon_idx]
        metrics, _, _ = naive.evaluate(y_test, horizon)
        naive_results[horizon] = metrics
        print(f"{horizon}h: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")
        horizon_idx += 1
    
    all_results['Naive'] = naive_results
    
    # train
    model_names = list(model_configs.keys())
    model_idx = 0
    
    while model_idx < len(model_names):
        model_name = model_names[model_idx]
        config = model_configs[model_name]
        
        print("\n" + "="*60)
        print(f"{model_name} Model")
        print("="*60)
        
        model_results = {}
        
        horizon_idx = 0
        while horizon_idx < len(horizons):
            horizon = horizons[horizon_idx]
            
            # Initial
            rnn = RNNClassifier(
                model_type=config['model_type'],
                sequence_length=sequence_length,
                hidden_units=config['hidden_units'],
                num_layers=config['num_layers'],
                dropout_rate=0.3,
                learning_rate=0.001,
                device=device
            )
            
            # Train
            history = rnn.train(
                X_train, y_train,
                X_val, y_val,
                horizon=horizon,
                epochs=epochs,
                batch_size=32
            )
            
            metrics, y_true, y_pred = rnn.evaluate(X_test, y_test, horizon)
            model_results[horizon] = metrics
            
            print(f"Test ({horizon}h): Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}, "
                  f"Prec={metrics['precision_macro']:.4f}, Rec={metrics['recall_macro']:.4f}")
            
            model_path = output_dir / "models" / f"{model_name.lower()}_h{horizon}.pt"
            rnn.save_model(model_path)
            
            # Plot
            plot_single_model_results(
                model_name, horizon, history, y_true, y_pred,
                output_dir / "plots"
            )
            
            horizon_idx += 1
        
        all_results[model_name] = model_results
        model_idx += 1

    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    
    # Comparison
    comparison_data = []
    model_names_list = list(all_results.keys())
    model_idx = 0
    
    while model_idx < len(model_names_list):
        model_name = model_names_list[model_idx]
        horizon_results = all_results[model_name]
        
        horizons_list = list(horizon_results.keys())
        horizon_idx = 0
        
        while horizon_idx < len(horizons_list):
            horizon = horizons_list[horizon_idx]
            metrics = horizon_results[horizon]
            
            comparison_data.append({
                'Model': model_name,
                'Horizon': f'{horizon}h',
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'F1 Score': f"{metrics['f1_macro']:.4f}",
                'Precision': f"{metrics['precision_macro']:.4f}",
                'Recall': f"{metrics['recall_macro']:.4f}"
            })
            
            horizon_idx += 1
        
        model_idx += 1
    
    comparison_df = pd.DataFrame(comparison_data)

    comparison_path = output_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
    
    plot_model_comparison(all_results, output_dir / "plots")
    
    # Save
    results_path = output_dir / "all_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    return all_results, comparison_df