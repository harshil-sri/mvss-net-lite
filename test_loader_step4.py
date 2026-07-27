from data_pipeline.dataset_loader import get_dataloader

def test_step4():
    train_loader, val_loader, test_loader = get_dataloader(
        ['CASIAv2', 'DEFACTO'], 
        batch_size=8, 
        is_train=True, 
        return_splits=True
    )
    
    # We can check len(loader.dataset) because it is a DataLoader and dataset is ForgeryDataset
    print(f"Train batches: {len(train_loader)} (Samples: {len(train_loader.dataset)})")
    print(f"Val batches:   {len(val_loader)} (Samples: {len(val_loader.dataset)})")
    print(f"Test batches:  {len(test_loader)} (Samples: {len(test_loader.dataset)})")
    
    # Confirm they only contain CASIAv2 and DEFACTO
    def check_dataset_paths(loader, name):
        ds = loader.dataset
        bad_paths = [p for p, _ in ds.samples if 'CASIAv2' not in p and 'DEFACTO' not in p]
        if bad_paths:
            print(f"ERROR: {name} contains {len(bad_paths)} non-CASIAv2/DEFACTO paths! e.g. {bad_paths[0]}")
        else:
            print(f"{name} cleanly isolated to requested datasets.")
            
    check_dataset_paths(train_loader, "Train Loader")
    check_dataset_paths(val_loader, "Val Loader")
    check_dataset_paths(test_loader, "Test Loader")

if __name__ == '__main__':
    test_step4()
