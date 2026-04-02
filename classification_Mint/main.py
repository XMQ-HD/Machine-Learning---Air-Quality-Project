import argparse
from experiments import run_classification_experiments


def main():
    parser = argparse.ArgumentParser(
        description='RNN Air Quality Classification - COMP9417 Project (PyTorch)'
    )
    parser.add_argument(
        '--data_dir', 
        default='../clean_data/airquality_prepared', 
        help='Data directory path'
    )
    parser.add_argument(
        '--output_dir', 
        default='rnn_classification_results_pytorch', 
        help='Output directory path'
    )
    parser.add_argument(
        '--sequence_length', 
        type=int, 
        default=24, 
        help='Input sequence length (hours)'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=50, 
        help='Number of training epochs'
    )
    parser.add_argument(
        '--device', 
        default=None,
        help='Device (cuda or cpu)'
    )
    
    args = parser.parse_args()
    
    # start experiment
    results, comparison = run_classification_experiments(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        device=args.device
    )
    
    print("\n" + "="*60)
    print("Top 5 Results:")
    print("="*60)
    print(comparison.head())


if __name__ == "__main__":
    main()