import pandas as pd
import numpy as np 
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import SaltRemover
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from scipy.stats import pearsonr, zscore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def random_split(data, train_ratio, test_ratio, seed):
    '''
    Performs random splitting based on a given ratio for the training, validation, and test sets. 
        data: file that contains the dataset
            (type: pandas dataframe or csv)
        train_ratio: size of the training set
            (type: float)
        test_ratio : size of the test set 
            (type: float)
        seed: the seed you want each run to hold
            (type: int)
        Returns: 3 pandas dataframe 
        
        example for an 8:1:1 splitting:
            train, val, test = random_split(dataset, 0.8, 0.1, 42)
        *This is a slightly modified version of sklearn's train_test_split function*
    '''
    np.random.seed(seed)
    shuffle_data = np.random.permutation(len(data))
    train_indices = shuffle_data[:int(len(data)*train_ratio)]
    val_indices = shuffle_data[int(len(data)*train_ratio):int(len(data)*(1.0-test_ratio))]
    test_indices = shuffle_data[int(len(data)*(1.0-test_ratio)):]
    return data.iloc[train_indices], data.iloc[val_indices], data.iloc[test_indices]

def make_canonical_smiles(df, molecule_column):
    '''
    Will return canonical smiles from a given dataset that contains SMILES string, since T5Chem reads in canonical smiles. 
        df: pandas dataframe with data that includes as minimum a column with molecules 
            (type: pandas dataframe)
        molecule_column: string of the name of the column that has the SMILES 
            (type: string)
    '''
    molecules = df[molecule_column].values
    cano_smiles = []
    #if molecules.str.contains('acidic:|basic:').any():
    if any('acidic:' in mol or 'basic:' in mol for mol in molecules):
        df[['prefix', 'smiles']] = df.smiles.str.split(":", expand = True)
        molecules = df['smiles'].values
        for mol in molecules:
            m = Chem.MolFromSmiles(mol)
            cano_smiles.append(Chem.MolToSmiles(m))
        df['canonical_smiles'] = cano_smiles
        df = df.drop(['smiles'], axis=1)
        df['canonical_smiles'] = df['prefix'].astype(str) + ':' + df['canonical_smiles']
        df = df.drop(['prefix'], axis=1)
        cols = list(df)
        cols.insert(0, cols.pop(cols.index('canonical_smiles')))
        df = df.loc[:, cols]
    else:
        for mol in molecules:
            m = Chem.MolFromSmiles(mol)
            cano_smiles.append(Chem.MolToSmiles(m))
        df['canonical_smiles'] = cano_smiles
        df = df.drop([molecule_column], axis=1)
        cols = list(df)
        cols.insert(0, cols.pop(cols.index('canonical_smiles')))
        df = df.loc[:, cols]
    return df 

def smiles_to_sdf(csv, path_to_save_sdf):
    '''
    Converts SMILES into SDF format. 
        csv: path of where the file that has smiles in is
            (type: string)
        path_to_save_df: path to where you want resulting SDF file in 
            (type: string)
    '''
    data = pd.read_csv(csv, names = ['smiles'])
    data['rdkit object'] = data['smiles'].apply(Chem.MolFromSmiles)
    PandasTools.WriteSDF(data, path_to_save_sdf, molColName='rdkit object', properties=list(data.columns))

def sdf_to_smiles(sdf):
    '''
    Turns an SDF files into a pandas dataframe that contains a column with SMILES 
        sdf: path to the sdf file
            (type: string)
    '''
    from rdkit.Chem.PandasTools import LoadSDF
    df = LoadSDF(sdf, smilesName='smiles')
    return df

def combine_scaffold_and_random_predictions(random_average, scaffold_average):
    '''
    To be used after evaluating ensemble model using ensemble_prediction.py. There should be two csv files: 
    one containing the averages and individual predictions from the random splitting and the other containing 
    the averages and individual predictions from scaffold splitting. This function will combine the predictions
    from models trained on random splitting and models trained from scaffold splitting to then perform the final
    evaluate of our final ensemble model. 
        random_average: path to csv file containing predictions of models trained on randomly split data 
            (type: string)
        scaffold_average: path to csv file containing predictions of models trained on scaffold split data
            (type: string)
        Returns: 1 pandas dataframe and 1 dictionary containing values used for evalution 

        example:
            df, eval_values = combine_scaffold_and_random_predictions('/path/to/csv_predictions_random', \
            '/path/to/csv_predictions_scaffold')
    '''
    rand = pd.read_csv(random_average)
    scaf = pd.read_csv(scaffold_average)
    rand = rand.drop(['average','STDev'], axis=1)
    for i in range(5):
        rand['prediction_'+str(i+5)] = scaf['prediction_'+str(i)]
    avg, std = [],[]
    for i in range(len(rand)):
        avg.append(rand[['prediction_0', 'prediction_1', 'prediction_2', 'prediction_3', 'prediction_4', \
        'prediction_5','prediction_6','prediction_7','prediction_8','prediction_9']].iloc[i].mean())
        std.append(rand[['prediction_0', 'prediction_1', 'prediction_2', 'prediction_3', 'prediction_4', \
        'prediction_5','prediction_6','prediction_7','prediction_8','prediction_9']].iloc[i].sem())
    rand['average'] = avg
    rand['STDev'] = std

    #evaluate
    r_value, prob = pearsonr(rand['targets'], rand['average'])
    rmse = mean_squared_error(rand['targets'], rand['average'], squared=False)
    mae = mean_absolute_error(rand['targets'], rand['average'])
    r2 = r2_score(rand['targets'], rand['average'])
    values = {'RMSE': rmse, 'MAE': mae, 'r2': r2, 'r': r_value}
    return rand, values

def plot_macropka_results(acid, basic=None, title=None):
    '''
    Plots a scatter plot for results model using macropka task. 
        acid: file with targets and predicted results for acidic molecules
            (type: pandas dataframe)
        basic: file with targets and predicted results for basic molecules 
            (type: pandas dataframe)
        title: name of plot
            (type: string)
        
        example: plot_macropka_results(acid_df, basic_df, 'Novartis Dataset')
    '''
    if basic is not None:
        nov_x = pd.concat([acid['targets'],basic['targets']], axis=0)
        nov_y = pd.concat([acid['average'],basic['average']], axis=0)
        plt.scatter(basic['targets'], basic['average'], c='orange', alpha=0.5, label='basic')
        plt.scatter(acid['targets'], acid['average'], c='cornflowerblue', alpha=0.6, label='acidic')
        mae = np.mean(np.abs(nov_x - nov_y))
        rmse = np.sqrt(np.mean((nov_x - nov_y) ** 2))
        r= pearsonr(nov_x, nov_y)[0]
        x_min = min(nov_x)
        x_max = max(nov_x)
        plt.plot([x_min, x_max], [x_min, x_max], color="black", linestyle="--")
        plt.xlabel("Target pka")
        plt.ylabel(f"Average predicted pka")
        if title is not None:
            plt.title(title)
        annotate = f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}\nPearson R = {r:.4f}\n"
        plt.annotate(annotate, xy=(0.23, 0.78), xycoords='axes fraction')
        plt.legend()
        plt.show()
    if basic == None:
        nov_x = acid['targets']
        nov_y = acid['average']
        plt.scatter(acid['targets'], acid['average'], c='cornflowerblue', alpha=0.6)
        mae = np.mean(np.abs(nov_x - nov_y))
        rmse = np.sqrt(np.mean((nov_x - nov_y) ** 2))
        r= pearsonr(nov_x, nov_y)[0]
    
        x_min = min(nov_x)
        x_max = max(nov_x)
        plt.plot([x_min, x_max], [x_min, x_max], color="black", linestyle="--")
        plt.xlabel("Target pka")
        plt.ylabel(f"Average predicted pka") 
        if title is not None:
            plt.title(title)
        annotate = f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}\nPearson R = {r:.4f}\n"
        plt.annotate(annotate, xy=(0.1, 0.78), xycoords='axes fraction')
        plt.legend()
        plt.show()

def bar_graph_acid_basic_ensemble(acid, basic):
    '''
    Creates a bar graph for results from macropka tasked model. 
        acid: file that contains targets and predicted results for acidic molecules
            (type: pandas dataframe)
        basic: file that contains targets and predicted results for basic molecules 
            (type: pandas dataframe)

        example: bar_graph_acid_basic_ensemble(acid_df, basic_df)
    '''
    bar_height = 0.25
    x_values = ['acidic', 'basic']
    y_pos = np.arange(len(x_values))
    
    rmse_values = [(np.sqrt(np.mean((acid['targets'] - acid['average']) ** 2))),(np.sqrt(np.mean((basic['targets'] - basic['average']) ** 2)))]
    mae_values = [np.mean(np.abs(acid['targets'] - acid['average'])), np.mean(np.abs(basic['targets'] - basic['average']))]
    r_values= [pearsonr(acid['targets'],acid['average'])[0],pearsonr(basic['targets'] ,basic['average'])[0]]
    
    fig, ax = plt.subplots(figsize=(6, 6))

    bars_rmse = ax.barh(y_pos - bar_height, rmse_values,bar_height, color=['cornflowerblue','orange'])
    bars_mae = ax.barh(y_pos , mae_values,bar_height,color=['cornflowerblue','orange'], hatch='--')
    bars_r = ax.barh(y_pos + bar_height , r_values, bar_height,color=['cornflowerblue','orange'], hatch='||')

    for i, bar in enumerate(bars_rmse):
        ax.annotate(f'RMSE: {rmse_values[i]:.3f}', xy=(bar.get_x() + bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0), textcoords='offset points', va='center', color='black', fontsize=10)
    for i, bar in enumerate(bars_mae):
        ax.annotate(f'MAE: {mae_values[i]:.3f}', xy=(bar.get_x() + bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0), textcoords='offset points', va='center', color='black', fontsize=10)
    for i, bar in enumerate(bars_r):
        ax.annotate(f'R: {r_values[i]:.3f}', xy=(bar.get_x() + bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0), textcoords='offset points', va='center', color='black', fontsize=10)
    # Set x-axis labels and title
    ax.set_xlabel('')
    ax.set_title('Individual Performance of acidic and basic molecules')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(x_values)
    ax.legend()

    # Remove spines (borders) from the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Adjust layout and display the plot
    plt.show()

def find_outliers(df, smiles_df, threshold_num):
    '''
    find the outlier from your csv that has predictive values. 
        df: file that contains targets and predictions
            (type: pandas dataframe)
        smiles_df: file that contains the SMILES feed into T5chem (ends in .source)
            (type: pandas dataframe)
        threshold_num: threshold (higher the number less outliers shown)
            (type: int)
        
        example: outliers_dict = find_outliers(prediction, smiles, 20)
    '''
    df['difference'] = df['target_0'] - df['prediction_0']
    df['z_scores'] = np.abs(zscore(df['difference']))
    df['smiles'] = smiles_df
    threshold = threshold_num
    outliers = df[df['z_scores'] > threshold]
    outliers = outliers.reset_index()
    
    d = {}
    for i in range(len(outliers)):
        index = outliers['index'].iloc[i]
        smiles = outliers['smiles'].iloc[i]
        true = outliers['target_0'].iloc[i]
        predicted = outliers['prediction_0'].iloc[i]
        d_ ={smiles:[index, true, predicted]}
        d.update(d_)
    return d

def remove_salts(smiles):
    mol = Chem.MolFromSmiles(smiles)
    remover = SaltRemover.SaltRemover()
    res = remover.StripMol(mol)
    s = Chem.MolToSmiles(res)
    return s
    
def get_atom_idx_based_on_charge(smiles):
    '''
    Still working on this function to be used in another script
    '''
    print(len(smiles))
    molecule = Chem.MolFromSmiles(smiles)
    for atom in molecule.GetAtoms():
        if atom.GetFormalCharge() != 0:
            print(atom.GetIdx())
            charged_molecule = Chem.PathToSubmol(molecule,[4,5,6,7,8])
            ringinfo = charged_molecule.GetRingInfo()
            Chem.GetSymmSSSR(charged_molecule, ringinfo=ringinfo)
            # Generate the Murcko scaffold for the charged molecule
            scaffold = MurckoScaffold.GetScaffoldForMol(charged_molecule, ringinfo=ringinfo)
            print(scaffold)
        
#get_atom_idx_based_on_charge('Clc1ccc(C[N-]c2ncnc3ccccc23)cc1')





