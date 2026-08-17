
import numpy as np
import pandas as pd
from anndata import AnnData
from cytof_helper.interactive import InteractiveClusterLabeler
import unittest
from unittest.mock import MagicMock

class TestInteractiveFinalize(unittest.TestCase):
    def test_finalize_assignments_full_data(self):
        # 1. Setup Data
        n_obs = 100
        n_vars = 5
        X = np.random.rand(n_obs, n_vars)
        obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
        var = pd.DataFrame(index=[f"marker_{i}" for i in range(n_vars)])
        obsm = {'X_umap': np.random.rand(n_obs, 2)}
        adata = AnnData(X=X, obs=obs, var=var, obsm=obsm)
        
        # 2. Initialize Labeler with subsampling
        labeler = InteractiveClusterLabeler(adata, subsample=50)
        
        # Verify subsampling happened
        self.assertLess(len(labeler.adata), len(labeler.adata_original))
        
        # 3. Simulate Training Results (Mocking _train_classifier effects)
        # Create fake predictions for the FULL dataset
        labeler.full_predicted_labels = np.array(['Cluster 1'] * n_obs)
        # Create probabilities: 
        # First 10: 0.9 (High confidence)
        # Next 10: 0.4 (Low confidence)
        # Rest: 0.9
        probs = np.ones(n_obs) * 0.9
        probs[10:20] = 0.4
        labeler.full_predicted_proba = probs
        
        # 4. Set Threshold
        labeler.prob_threshold = 0.5
        
        # 5. Call save_full_labels directly (mimic button click logic)
        labeler.save_full_labels()
        
        # 6. Verify Results in ORIGINAL AnnData
        saved_labels = adata.obs['predicted_labels']
        saved_probs = adata.obs['predicted_proba']
        
        # High confidence cells (index 0)
        self.assertEqual(saved_labels.iloc[0], 'Cluster 1')
        self.assertEqual(saved_probs.iloc[0], 0.9)
        
        # Low confidence cells (index 10)
        self.assertEqual(saved_labels.iloc[10], 'Uncertain')
        self.assertEqual(saved_probs.iloc[10], 0.4)
        
        print("Verification Successful: Full dataset updated correctly with threshold.")

if __name__ == '__main__':
    unittest.main()
