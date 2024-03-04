import pandas as pd
import argparse

'''
example input: python seq2seq_to_micropka.py --seq2seq_input ../../pka/data/SAMPL8/unified/seq2seq/test.source --seq2seq_output ../../model_DEPROT/mixed/finetune/sampl8.csv  \
--regression_targets ../../pka/data/SAMPL8/pairs/test.target --save ../../pka/data/SAMPL8/unified/micropka_regression/
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq2seq_input",
        type=str,
        required=True,
        help="path to directory that contains inputs of seq2seq model (.source file)",
    )
    parser.add_argument(
        "--seq2seq_output",
        type=str,
        required=True,
        help="path to directory that contains results from seq2seq model .csv file",
    )
    parser.add_argument(
        "--regression_targets",
        type=str,
        required=True,
        help="path to directory that contains targets needed in regression task .csv file",
    )
    parser.add_argument(
        "--save",
        type=str,
        required=True,
        help="path to directory to save .source and .target files for input into regression model",
    )

    args = parser.parse_args()

    result = pd.read_csv(args.seq2seq_output)
    inputs = pd.read_csv(args.seq2seq_input, names=['source'])
    targets = pd.read_csv(args.regression_targets, names=['target'])
    inputs[['prefix','smiles']] = inputs['source'].str.split(':',expand=True)

    pairs = []
    for i, (x,y,z) in enumerate(zip(inputs['smiles'], result['prediction_1'], result['prediction_2'])):
        if not pd.isna(y):
            pairs.append(str(x) + ">>" + str(y))
        else:
            pairs.append(str(x) + ">>" + str(z))
    
    df = pd.DataFrame({'source': pairs, 'target': targets['target']})

    df['source'].to_csv(str(args.save)+"test.source", index=False, header=False)
    df['target'].to_csv(str(args.save)+"test.target", index=False, header=False)
    
if __name__ == "__main__":
    main() 