"""Regression checks for ingestion, dataset merging, splits and playback export."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import load_datasets, stratified_split
import ingest_recordings
import export_playback

class PipelineTests(unittest.TestCase):
    def test_merging_remaps_labels_and_signer_groups(self):
        with tempfile.TemporaryDirectory() as root:
            paths=[]
            for i, labels in enumerate([['Hello','Doctor'],['Doctor','Thank you']]):
                p=Path(root)/f'{i}.npz'; paths.append(p)
                np.savez(p,X=np.zeros((4,32,65,3)), y=np.array([0,1,0,1]), signer=np.zeros(4,dtype=int),labels=np.array(labels))
            X,y,groups,labels,digest=load_datasets(paths)
            self.assertEqual(labels,['Doctor','Hello','Thank you'])
            self.assertEqual(y.tolist(),[1,0,1,0,0,2,0,2])
            self.assertEqual(groups.tolist(),[0]*4+[1]*4)
            self.assertEqual(len(digest),64);self.assertEqual(len(X),8)
    def test_stratified_train_validation_test_do_not_overlap(self):
        y=np.repeat(np.arange(5),20); rng=np.random.default_rng(2)
        tr,te=stratified_split(y,rng); fit,val=stratified_split(y[tr],rng)
        self.assertFalse(set(tr)&set(te));self.assertFalse(set(fit)&set(val))
        self.assertEqual(set(y[tr[fit]]),set(y));self.assertEqual(set(y[te]),set(y))
    def test_playback_export_covers_all_dataset_labels(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'dataset.npz';out=Path(root)/'signs.json'
            np.savez(path,X=np.ones((4,32,65,3)), y=np.array([0,1,0,1]), signer=np.zeros(4,dtype=int),labels=np.array(['Hello','Doctor']))
            with patch.object(sys,'argv',['export_playback','--data',str(path),'--own',str(Path(root)/'absent'),'--output',str(out)]):
                self.assertEqual(export_playback.main(),0)
            self.assertEqual(set(json.loads(out.read_text())),{'Hello','Doctor'})
    def test_ingestion_counts_only_valid_unique_takes_and_creates_data_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rng=np.random.default_rng(5);takes=[]
            for i in range(10):
                f=rng.uniform(.3,.6,(20,65,3));f[:,11,:2]=[.4,.3];f[:,12,:2]=[.6,.3]
                takes.append({'gloss':'Hello','frames':f.tolist()})
            bad=np.array(takes[0]['frames']);bad[:,23:65]=0
            takes.extend([takes[0],{'gloss':'Bad','frames':bad.tolist()}])
            source=root/'recordings.json';source.write_text(json.dumps({'format':'setu-recordings-v1','takes':takes}))
            out=root/'data/own.npz'
            with patch.object(ingest_recordings,'ROOT',root),patch.object(ingest_recordings,'OUT',out),patch.object(sys,'argv',['ingest',str(source)]):
                self.assertEqual(ingest_recordings.main(),0)
            with np.load(out) as d:self.assertEqual(len(d['X']),10);self.assertEqual(d['labels'].tolist(),['Hello'])
    def test_invalid_dataset_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'bad.npz'
            np.savez(p,X=np.full((4,32,65,3),np.nan),y=np.zeros(4,dtype=int),signer=np.zeros(4,dtype=int),labels=['Hello'])
            with self.assertRaises(ValueError):load_datasets([p])

if __name__=='__main__':unittest.main()
