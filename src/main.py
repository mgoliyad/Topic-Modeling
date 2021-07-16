import argparse
import json
from pathlib import Path
from topic_modeling import Topic_modeling


def main():

    """Main function."""
    
    condif_filename = Path(Path.cwd(), 'config.json')
    with open(condif_filename) as f:
        config = json.load(f)
        
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Directory containing texts.', type=Path, default=Path(Path.cwd().parent, 'CLEAN_DATA'))
    parser.add_argument('--output_dir', help='Directory to save XML for topic modeling.', type=Path, default=Path(Path.cwd().parent, 'OUTPUT'))
    parser.add_argument('--printout', help='Do you want to see printouts?', type=bool, default=True)
    
    args = parser.parse_args()
    printout = args.printout
    data_dir = args.data_dir
    output_dir = args.output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)

  
    T_classifier = Topic_modeling(output_dir, config, printout)
    T_classifier.gensim_LDA(data_dir)

    
if __name__ == '__main__':
    main()
    print('Press any key to exit the program...')
    input()
