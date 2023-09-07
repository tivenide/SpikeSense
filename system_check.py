def system_check():
    """
    Prints available resources.
    :return: None
    """
    import torch
    import os

    print('Total CPUs (threads) found:\t\t',os.cpu_count())
    print('Available threads for torch:\t',torch.get_num_threads())

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print('Total GPUs found:\t\t\t\t',str(count))
        for i in range(torch.cuda.device_count()):
            print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i))
    else:
        print('No GPUs found!')
