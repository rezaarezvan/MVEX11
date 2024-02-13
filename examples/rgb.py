import torch
import torch.nn.functional as F

NUM_IMAGES = 10


def softmax_attention(Q, K, V, d_model):
    attention_scores = torch.matmul(Q, K.transpose(-2, -1))/torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
    attention_weights = F.softmax(attention_scores, dim=-1)
    return torch.matmul(attention_weights, V), attention_weights


def find_max_red_pic(pics):
    red_values = pics[:, 0]
    max_red_index = torch.argmax(red_values)
    return max_red_index


def main():
    # RGB channel values for num_images images
    R = torch.rand(NUM_IMAGES)  # Red channel values
    G = torch.rand(NUM_IMAGES)  # Green channel values
    B = torch.rand(NUM_IMAGES)  # Blue channel values

    pics = torch.stack([R, G, B], dim=1)  # Shape: [num_images, 3]

    print(f"Image (RGB Values): \n{pics.numpy()}")
    print(f"Image shape: {pics.numpy().shape}")

    # Keys (K) and Values (V) are the same as the input image RGB values
    K = V = pics
    print(f"Keys (K): \n{K.numpy()}")
    print(f"Values (V): \n{V.numpy()}")
    print(f"Keys shape: {K.numpy().shape}")
    print(f"Values shape: {V.numpy().shape}")

    # Query (Q) is a single vector to find "red" in the image
    Q = torch.tensor([[1.0, 0.0, 0.0]])
    print(f"Query (Q): {Q.numpy()}")
    print(f"Query shape: {Q.numpy().shape}")

    # Dimensionality of the input image (3 for RGB)
    d_model = 3

    # Compute attention and obtain weights
    attended_values, attention_weights = softmax_attention(Q, K, V, d_model)
    print(f"Attended values: \n{attended_values.numpy()}")
    print(f"Attention weights: \n{attention_weights.numpy()}")

    # Find the image with the maximum red value
    groundtruth_max_red_index = find_max_red_pic(pics)
    print(f"Image with maximum red value: {groundtruth_max_red_index}")

    # Image with most redness according to the attention weights
    attention_max_red_index = torch.argmax(attention_weights)
    print(f"Image with maximum red value according to attention: {attention_max_red_index}")


if __name__ == "__main__":
    main()
