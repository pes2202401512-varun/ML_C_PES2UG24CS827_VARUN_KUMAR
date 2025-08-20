import torch

# -------------------------------
# 1. Entropy of Dataset
# -------------------------------
def get_entropy_of_dataset(tensor: torch.Tensor) -> float:
    """
    Calculates entropy of the dataset
    tensor: torch.Tensor with last column as target
    """
    target_col = tensor[:, -1]   # last column = class labels
    classes, counts = torch.unique(target_col, return_counts=True)
    probs = counts.float() / counts.sum()

    entropy = -torch.sum(probs * torch.log2(probs))
    return entropy.item()


# -------------------------------
# 2. Average Info of an Attribute
# -------------------------------
def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int) -> float:
    """
    Calculates average information (expected entropy) for a given attribute
    attribute: column index to split on
    """
    attr_col = tensor[:, attribute]
    classes, counts = torch.unique(attr_col, return_counts=True)

    total_samples = tensor.shape[0]
    avg_info = 0.0

    for i, cls in enumerate(classes):
        subset = tensor[attr_col == cls]
        weight = counts[i].item() / total_samples
        entropy_subset = get_entropy_of_dataset(subset)
        avg_info += weight * entropy_subset

    return avg_info


# -------------------------------
# 3. Information Gain
# -------------------------------
def get_information_gain(tensor: torch.Tensor, attribute: int) -> float:
    """
    Info Gain = Entropy(D) - AvgInfo(attribute)
    """
    dataset_entropy = get_entropy_of_dataset(tensor)
    avg_info = get_avg_info_of_attribute(tensor, attribute)
    return dataset_entropy - avg_info


# -------------------------------
# 4. Best Attribute Selection
# -------------------------------
def get_selected_attribute(tensor: torch.Tensor):
    """
    Returns dict of {attribute: information_gain} and selected attribute
    """
    n_attributes = tensor.shape[1] - 1   # exclude target
    info_gains = {}

    for attr in range(n_attributes):
        ig = get_information_gain(tensor, attr)
        info_gains[attr] = ig

    # Pick attribute with max info gain
    selected_attribute = max(info_gains, key=info_gains.get)
    return (info_gains, selected_attribute)
