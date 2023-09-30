def development_workflow_quick():
    import torch
    import torch.nn as nn

    from utilities import prepro

    from ml_framework import create_dataloader_simple
    from ml_framework import load_datasets_from_h5
    from ml_framework import EarlyStopper
    from ml_framework import modeling

    from ml_models import DenseModel

    print('SpikeSense development started (quick)')

    # paths
    path_to_original_recordings = 'quick/MEArec_training_data_recordings'
    path_to_working_directory = 'quick/working_directory'
    path_to_saving_resulting_dataset = 'quick/working_directory/dataset.h5'
    dataset_name = 'dataset.h5'

    # preprocessing
    cur_prepro = prepro(path_to_original_recordings=path_to_original_recordings,
                        path_to_target_prepro_results=path_to_saving_resulting_dataset, threshold=100,
                        windowsize_in_sec=0.002, step_size=None, feature_calculation=False, scaler_type=None,
                        sampling_strategy='majority')
    cur_prepro.run()

    # data loading
    x_train, y_train, x_test, y_test, x_val, y_val = load_datasets_from_h5(path_to_saving_resulting_dataset)

    batch_size = 32

    train_dataloader = create_dataloader_simple(x_train, y_train, batch_size=batch_size)
    eval_dataloader = create_dataloader_simple(x_val, y_val, batch_size=batch_size)
    test_dataloader = create_dataloader_simple(x_test, y_test, batch_size=batch_size)

    # models
    fnn = DenseModel(in_features=25, hidden_features=50, out_features=2)

    # model selection
    model = fnn

    # training definition
    torch.manual_seed(0)

    lr = 1e-4
    wd = 1e-5
    epochs = 5

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    early_stop = EarlyStopper(patience=15, min_delta=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)

    trainer_metadata = {
        'batch_size': batch_size,
        'optimizer': 'Adam',
        'learning_rate': lr,
        'weight_decay': wd,
        'loss_fn': str(loss_fn)
    }

    # run
    cur_model = modeling(model=model,
                         train_dataloader=train_dataloader, eval_dataloader=eval_dataloader,
                         test_dataloader=test_dataloader,
                         epochs=epochs, loss_fn=loss_fn, early_stop=early_stop, optimizer=optimizer,
                         scheduler=scheduler, n_classes=2)
    cur_model.run_and_save_best_and_last_model(path_to_working_directory, dataset_name)
    cur_model.save_trainer_meta_to_h5(path_to_saving_resulting_dataset, trainer_metadata)

    cur_model.plot_metrics_over_epochs(cur_model.train_some_metrics, title='Train metrics over epochs')
    cur_model.plot_metrics_over_epochs(cur_model.eval_some_metrics, title='Eval metrics over epochs')
    cur_model.plot_confusion_matrix(cur_model.test_some_metrics['confusion_matrix'], ['Class 0', 'Class 1'])

    print('SpikeSense development finished (quick)')

