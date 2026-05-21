#!/usr/bin/env python3
# This script needs ~12GB of RAM to run (because of the images).
import datasets
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

SUBSETS = ['text-CZ', 'text-SK', 'text-UA', 'visual-CZ', 'visual-SK', 'visual-UA']
SPLITS = ['dev', 'test']

EN_REGIONS = {
    'CZ': 'en_CZ',
    'SK': 'en_SK',
    'UA': 'en_UA'
}
LOC_REGIONS = {
    'CZ': 'cs_CZ',
    'SK': 'sk_SK',
    'UA': 'uk_UA'
}

def combine(split: str) -> datasets.Dataset:
    new_ds = {
        'id': [],
        'split': [],        
        'modality': [],
        'region': [],
        'question': [],
        'answer': [],
        'image': []
    }
    for subset in SUBSETS:
        # Build row by row
        ds = datasets.load_dataset('ufal/cus-qa', subset, split=split)
        for row in tqdm(ds):
            # Local
            new_ds['id'].append(row['id'])
            new_ds['split'].append(split)
            new_ds['modality'].append(subset.split('-')[0])
            new_ds['region'].append(LOC_REGIONS[subset.split('-')[1]])
            new_ds['question'].append(row['question_orig'])
            new_ds['answer'].append(row.get('answer_orig', None))
            new_ds['image'].append(row.get('image', None))
            
            # English
            new_ds['id'].append(row['id'])
            new_ds['split'].append(split)
            new_ds['modality'].append(subset.split('-')[0])
            new_ds['region'].append(EN_REGIONS[subset.split('-')[1]])
            new_ds['question'].append(row['question_en'])
            new_ds['answer'].append(row.get('answer_en', None))
            new_ds['image'].append(row.get('image', None))
    new_ds = datasets.Dataset.from_dict(new_ds)
    print('Sorting...')
    new_ds = new_ds.sort(['region', 'modality', 'id']) # sort by 'block'
    return new_ds

def to_json(ds: datasets.Dataset, path: str) -> None:    
    ds = ds.remove_columns('image')
    ds.to_json(path)
    

if __name__ == '__main__':
    print('Preparing the dev set...')
    dev_ds = combine('dev')
    print('Saving...')
    dev_ds.save_to_disk('cus_qa_dev')
    to_json(dev_ds, 'cus_qa_dev.jsonl')

    print('Preparing the test set...')
    test_ds = combine('test')    
    print('Saving...')
    test_ds.save_to_disk('cus_qa_test')
    to_json(test_ds, 'cus_qa_test.jsonl')
    
