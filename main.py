

if __name__ == '__main__':
    print('SpikeSense started')

    import torch
    import torch.nn as nn

    from utilities import load_frame_from_disk
    from ai_modeling import create_dataloader_simple

    from ai_modeling import EarlyStopper
    from ai_modeling import modeling

    from custom_models import DenseModel, TransformerModel

    train_dataloader = create_dataloader_simple(load_frame_from_disk('/mnt/MainNAS/BioMemsLaborNAS/Projekt_Ordner/STRIPE/data/prepared_for_training/frames_x_train_res.npy'), load_frame_from_disk('/mnt/MainNAS/BioMemsLaborNAS/Projekt_Ordner/STRIPE/data/prepared_for_training/frames_y_train_res.npy'))
    eval_dataloader = create_dataloader_simple(load_frame_from_disk('/mnt/MainNAS/BioMemsLaborNAS/Projekt_Ordner/STRIPE/data/prepared_for_training/frames_x_val_crp.npy'), load_frame_from_disk('/mnt/MainNAS/BioMemsLaborNAS/Projekt_Ordner/STRIPE/data/prepared_for_training/frames_y_val_crp.npy'))
    test_dataloader = create_dataloader_simple(load_frame_from_disk('/mnt/MainNAS/BioMemsLaborNAS/Projekt_Ordner/STRIPE/data/prepared_for_training/frames_x_test_crp.npy'), load_frame_from_disk('/mnt/MainNAS/BioMemsLaborNAS/Projekt_Ordner/STRIPE/data/prepared_for_training/frames_y_test_crp.npy'))

    dm = DenseModel(in_features=20, hidden_features=50, out_features=2)
    tm = TransformerModel(input_dim=1, hidden_size=64, num_classes=2, num_layers=12, num_heads=8, dropout=0.1)

    model = dm
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    early_stop = EarlyStopper(patience=15, min_delta=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)

    torch.manual_seed(0)

    cur_model = modeling(model=model,
                         train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, test_dataloader=test_dataloader,
                         epochs=6, loss_fn=loss_fn, early_stop=early_stop, optimizer=optimizer, scheduler=scheduler, n_classes=2, device=device)
    cur_model.run_and_save_best_and_last_model()



    print('SpikeSense finished')


