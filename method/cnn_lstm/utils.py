import numpy as np


def statistics(train_content_list, test_content_list):
    length = np.array([len(tmp) for tmp in train_content_list])
    print('Max train length:', np.max(length))
    print('Min train length:', np.min(length))
    print(f"Mean train length: {np.mean(length):.2f}")
    test_length = np.array([len(tmp) for tmp in test_content_list])
    print('Max test length:', np.max(test_length))
    print('Min test length:', np.min(test_length))
    print(f"Mean test length: {np.mean(test_length):.2f}")

    