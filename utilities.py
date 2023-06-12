# save to and load from disk
def save_frame_to_disk(frame, path_target):
    import numpy as np
    print('started saving frame to disk')
    np.save(path_target, frame, allow_pickle=True)
    print('frame saved to disk')


def load_frame_from_disk(path_source):
    import numpy as np
    print('started loading frame from disk')
    frame = np.load(path_source, allow_pickle=True)
    print('frame loaded from disk')
    return frame